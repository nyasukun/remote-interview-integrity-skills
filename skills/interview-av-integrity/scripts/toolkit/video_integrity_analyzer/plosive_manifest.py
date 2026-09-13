from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_P_KANA = frozenset("パピプペポぱぴぷぺぽ")
_B_KANA = frozenset("バビブベボばびぶべぼ")
_IGNORED_FOR_POSITION = frozenset(" \t\r\n、。,.!?！？「」『』（）()[]【】")


@dataclass(frozen=True)
class SpeakerInterval:
    epoch_id: str
    speaker: str
    group: str
    start_s: float
    end_s: float
    source: str = "stable active-speaker label"


@dataclass(frozen=True)
class BilabialToken:
    event_id: str
    speaker: str | None
    group: str | None
    epoch_id: str | None
    phoneme_class: str
    kana: str
    token_text: str
    token_start_s: float
    token_end_s: float
    anchor_s: float
    asr_probability: float
    segment_id: int | None
    segment_no_speech_probability: float | None
    eligible: bool
    exclusion_reason: str | None
    reading_text: str | None = None
    reading_source: str = "surface_word"
    # Rough ASR spans of the neighbouring words (transcript order). The
    # acoustic estimator uses them only to attribute burst candidates to the
    # target word or a neighbour; they are never treated as release times.
    previous_word_window_s: tuple[float, float] | None = None
    next_word_window_s: tuple[float, float] | None = None
    # Position of this kana among all bilabial kana of the same ASR word, and
    # the rough proportional anchors of all of them, so one burst is never
    # measured twice for one word.
    word_occurrence_index: int = 0
    word_occurrence_anchors_s: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path_or_data: str | Path | Mapping[str, Any]) -> Any:
    if isinstance(path_or_data, Mapping):
        return path_or_data
    return json.loads(Path(path_or_data).read_text(encoding="utf-8"))


def load_speaker_intervals(
    path_or_data: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[SpeakerInterval]:
    if isinstance(path_or_data, Sequence) and not isinstance(path_or_data, (str, bytes, Path)):
        raw: Iterable[Mapping[str, Any]] = path_or_data
    else:
        data = _read_json(path_or_data)  # type: ignore[arg-type]
        raw = data if isinstance(data, list) else data.get("intervals", ())
    intervals: list[SpeakerInterval] = []
    for item in raw:
        interval = SpeakerInterval(
            epoch_id=str(item["epoch_id"]),
            speaker=str(item["speaker"]),
            group=str(item["group"]),
            start_s=float(item["start_s"]),
            end_s=float(item["end_s"]),
            source=str(item.get("source", "stable active-speaker label")),
        )
        if not (math.isfinite(interval.start_s) and math.isfinite(interval.end_s)):
            raise ValueError(f"non-finite interval: {interval.epoch_id}")
        if interval.start_s < 0 or interval.end_s <= interval.start_s:
            raise ValueError(f"invalid interval: {interval.epoch_id}")
        intervals.append(interval)
    intervals.sort(key=lambda value: (value.start_s, value.end_s, value.epoch_id))
    for previous, current in zip(intervals, intervals[1:]):
        if current.start_s < previous.end_s - 1e-9:
            raise ValueError(
                f"speaker intervals overlap: {previous.epoch_id} and {current.epoch_id}"
            )
    return intervals


def _position_characters(text: str) -> list[tuple[int, str]]:
    normalized = unicodedata.normalize("NFKC", text)
    return [
        (source_index, character)
        for source_index, character in enumerate(normalized)
        if character not in _IGNORED_FOR_POSITION
    ]


def _anchor_for_character(text: str, source_index: int, start_s: float, end_s: float) -> float:
    characters = _position_characters(text)
    if not characters or end_s <= start_s:
        return start_s
    rank = next(
        (index for index, (original_index, _) in enumerate(characters) if original_index == source_index),
        0,
    )
    # Whisper word timestamps bound the complete token. Use the midpoint of the
    # target character's proportional slot as the rough anchor. Conservative
    # offset gates assume this convention as an uncalibrated starting point;
    # waveform analysis later refines the release itself.
    return start_s + (end_s - start_s) * (rank + 0.5) / len(characters)


def _assign_interval(
    anchor_s: float,
    intervals: Sequence[SpeakerInterval],
    *,
    boundary_guard_s: float,
) -> tuple[SpeakerInterval | None, str | None]:
    raw_match = next(
        (interval for interval in intervals if interval.start_s <= anchor_s < interval.end_s),
        None,
    )
    if raw_match is None:
        return None, "outside_labeled_speaker_interval"
    if (
        anchor_s < raw_match.start_s + boundary_guard_s
        or anchor_s >= raw_match.end_s - boundary_guard_s
    ):
        return raw_match, "near_active_speaker_transition"
    return raw_match, None


def extract_bilabial_tokens(
    whisper_data: str | Path | Mapping[str, Any],
    intervals: Sequence[SpeakerInterval],
    *,
    interview_end_s: float,
    minimum_asr_probability: float = 0.75,
    maximum_segment_no_speech_probability: float = 1.0,
    boundary_guard_s: float = 0.75,
) -> list[BilabialToken]:
    """Preselect every explicit /p/ and /b/ kana before inspecting visual lag.

    A word-level ``reading``/``kana``/``pronunciation`` field is preferred over
    the orthographic surface form. Exhaustive Japanese phoneme inventory is
    only supportable when upstream normalization supplies readings for every
    lexical word. Excluded tokens are returned alongside eligible ones.
    """
    data = _read_json(whisper_data)
    events: list[BilabialToken] = []
    serial = 0
    # Flatten words in transcript order so neighbour spans cross segment
    # boundaries; a pause-separated neighbour still bounds attribution.
    all_words: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = [
        (segment, word)
        for segment in data.get("segments", ())
        for word in segment.get("words", ())
    ]
    for word_index, (segment, word) in enumerate(all_words):
        segment_id = segment.get("id")
        no_speech = segment.get("no_speech_prob")
        no_speech_value = float(no_speech) if no_speech is not None else None
        previous_window = _word_window(
            all_words[word_index - 1][1] if word_index > 0 else None
        )
        next_window = _word_window(
            all_words[word_index + 1][1] if word_index + 1 < len(all_words) else None
        )
        text = unicodedata.normalize("NFKC", str(word.get("word", "")))
        reading_source = next(
            (
                key
                for key in ("reading", "kana", "pronunciation")
                if str(word.get(key, "")).strip()
            ),
            None,
        )
        reading = unicodedata.normalize(
            "NFKC",
            str(word.get(reading_source, "")) if reading_source else text,
        )
        try:
            start_s = float(word["start"])
            end_s = float(word["end"])
            probability = float(word.get("probability", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        bilabial_positions = [
            source_index
            for source_index, kana in enumerate(reading)
            if kana in _P_KANA or kana in _B_KANA
        ]
        occurrence_anchors = tuple(
            _anchor_for_character(reading, source_index, start_s, end_s)
            for source_index in bilabial_positions
        )
        for occurrence_index, source_index in enumerate(bilabial_positions):
            kana = reading[source_index]
            phoneme_class = "p" if kana in _P_KANA else "b"
            serial += 1
            anchor_s = occurrence_anchors[occurrence_index]
            interval, interval_exclusion = _assign_interval(
                anchor_s,
                intervals,
                boundary_guard_s=boundary_guard_s,
            )
            reasons: list[str] = []
            if anchor_s >= interview_end_s:
                reasons.append("after_interview_cutoff")
            if probability < minimum_asr_probability:
                reasons.append("low_asr_probability")
            if (
                no_speech_value is not None
                and no_speech_value > maximum_segment_no_speech_probability
            ):
                reasons.append("high_segment_no_speech_probability")
            if interval_exclusion:
                reasons.append(interval_exclusion)
            events.append(
                BilabialToken(
                    event_id=f"pb-{serial:04d}",
                    speaker=interval.speaker if interval else None,
                    group=interval.group if interval else None,
                    epoch_id=interval.epoch_id if interval else None,
                    phoneme_class=phoneme_class,
                    kana=kana,
                    token_text=text,
                    token_start_s=start_s,
                    token_end_s=end_s,
                    anchor_s=anchor_s,
                    asr_probability=probability,
                    segment_id=int(segment_id) if segment_id is not None else None,
                    segment_no_speech_probability=no_speech_value,
                    eligible=not reasons,
                    exclusion_reason=";".join(reasons) if reasons else None,
                    reading_text=reading,
                    reading_source=reading_source or "surface_word",
                    previous_word_window_s=previous_window,
                    next_word_window_s=next_window,
                    word_occurrence_index=occurrence_index,
                    word_occurrence_anchors_s=occurrence_anchors,
                )
            )
    return events


def _word_window(word: Mapping[str, Any] | None) -> tuple[float, float] | None:
    """Return a usable ``(start_s, end_s)`` span for a neighbouring ASR word.

    Zero-length, reversed, negative, or non-finite spans are returned as
    ``None`` so they can neither confer nor block attribution.
    """

    if word is None:
        return None
    try:
        start_s = float(word["start"])
        end_s = float(word["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        return None
    if start_s < 0.0 or end_s <= start_s:
        return None
    return (start_s, end_s)
