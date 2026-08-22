from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import mediapipe as mp
import numpy as np

from .models import FaceDetection, FaceSample, FaceTrack


# MediaPipe's 478-point face mesh indices.  The pairs follow the inner lip
# contours from the left side of the image to the right side.  Aperture is
# measured perpendicular to the mouth-corner axis, making it insensitive to
# in-plane head roll.
_MOUTH_CORNERS = (61, 291)
_INNER_LIP_PAIRS = (
    (80, 88),
    (81, 178),
    (82, 87),
    (13, 14),
    (312, 317),
    (311, 402),
    (310, 318),
)
_INNER_LIP_WEIGHTS = np.asarray((0.55, 0.80, 1.00, 1.25, 1.00, 0.80, 0.55))

# The three central pairs are the most useful for visible bilabial contact.
# Keeping all three, rather than only landmark 13–14, exposes asymmetric or
# unreliable fits that can otherwise look like a closed mouth numerically.
_CENTRAL_LIP_PAIRS = ((13, 14), (82, 87), (312, 317))

# A stable, ordered lip-shape descriptor.  Translation, roll, and size are
# removed before these x/y coordinates are stored in FaceSample.
_LIP_SHAPE_INDICES = (
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    intersection_w = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    intersection_h = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    intersection = intersection_w * intersection_h
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union > 1e-12 else 0.0


def _bbox_diagonal(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return math.hypot(x1 - x0, y1 - y0)


def _gradient_energy(rgb_patch: np.ndarray) -> float:
    """Return a robust 8-bit spatial-gradient magnitude for a color patch."""
    if rgb_patch.ndim != 3 or min(rgb_patch.shape[:2]) < 3:
        return math.nan
    patch = rgb_patch.astype(np.float32, copy=False)
    gray = 0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2]
    horizontal = np.abs(np.diff(gray, axis=1)).ravel()
    vertical = np.abs(np.diff(gray, axis=0)).ravel()
    if horizontal.size + vertical.size == 0:
        return math.nan
    # The mean is useful for compressed video, while clipping suppresses a few
    # subtitle/tile-border edges that would otherwise dominate the metric.
    gradients = np.concatenate((horizontal, vertical))
    clip_at = float(np.percentile(gradients, 95.0))
    return float(np.mean(np.minimum(gradients, clip_at)))


def _bilinear_rgb(rgb: np.ndarray, sample_x: np.ndarray, sample_y: np.ndarray) -> np.ndarray:
    """Sample RGB coordinates; callers must first check that they are in bounds."""
    height, width = rgb.shape[:2]
    x = np.clip(sample_x, 0.0, width - 1.001)
    y = np.clip(sample_y, 0.0, height - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    top = rgb[y0, x0].astype(np.float64) * (1.0 - wx) + rgb[y0, x1].astype(np.float64) * wx
    bottom = rgb[y1, x0].astype(np.float64) * (1.0 - wx) + rgb[y1, x1].astype(np.float64) * wx
    return top * (1.0 - wy) + bottom * wy


def _appearance_descriptor(full_rgb: np.ndarray, points: np.ndarray) -> tuple[float, ...]:
    """Build a compact aligned appearance descriptor, not an identity embedding.

    It is used only to keep speaker tiles from being merged at a hard layout
    cut.  The vector combines normalized low-resolution luminance with coarse
    color histograms and is intentionally much simpler than face recognition.
    """
    left_eye = points[33]
    right_eye = points[263]
    eye_axis = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_axis))
    if not np.isfinite(eye_distance) or eye_distance < 12.0:
        return ()
    axis_x = eye_axis / eye_distance
    axis_y = np.asarray((-axis_x[1], axis_x[0]), dtype=np.float64)
    eye_midpoint = (left_eye + right_eye) * 0.5

    # The canonical grid covers forehead through chin while staying mostly
    # inside the face oval.  Alignment reduces sensitivity to roll and zoom.
    horizontal = np.linspace(-1.0, 1.0, 10, dtype=np.float64)
    vertical = np.linspace(-0.62, 1.48, 10, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(horizontal, vertical)
    sample_x = eye_midpoint[0] + eye_distance * (grid_x * axis_x[0] + grid_y * axis_y[0])
    sample_y = eye_midpoint[1] + eye_distance * (grid_x * axis_x[1] + grid_y * axis_y[1])
    height, width = full_rgb.shape[:2]
    valid = (
        (sample_x >= 0.0)
        & (sample_x < width - 1)
        & (sample_y >= 0.0)
        & (sample_y < height - 1)
    )
    if float(np.mean(valid)) < 0.90:
        return ()
    canonical = _bilinear_rgb(full_rgb, sample_x, sample_y) / 255.0
    luminance = (
        0.2126 * canonical[..., 0]
        + 0.7152 * canonical[..., 1]
        + 0.0722 * canonical[..., 2]
    )
    luminance_std = float(np.std(luminance))
    if luminance_std < 0.015:
        return ()
    texture = np.clip((luminance - np.mean(luminance)) / max(luminance_std, 0.04), -2.5, 2.5).ravel()
    texture /= max(float(np.linalg.norm(texture)), 1e-12)

    histogram_parts: list[np.ndarray] = []
    for channel in range(3):
        histogram, _ = np.histogram(canonical[..., channel], bins=8, range=(0.0, 1.0))
        histogram = np.sqrt(histogram.astype(np.float64) / canonical[..., channel].size)
        histogram_parts.append(histogram)
    histogram_vector = np.concatenate(histogram_parts)
    histogram_vector /= max(float(np.linalg.norm(histogram_vector)), 1e-12)
    descriptor = np.concatenate((0.82 * texture, 0.48 * histogram_vector))
    descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
    return tuple(float(value) for value in descriptor)


def _appearance_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denominator <= 1e-12:
        return None
    similarity = float(np.dot(left_values, right_values) / denominator)
    return 1.0 - float(np.clip(similarity, -1.0, 1.0))


def _extract_patch(
    rgb: np.ndarray,
    bounds_px: tuple[float, float, float, float],
) -> np.ndarray | None:
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = bounds_px
    ix0 = max(0, min(width, int(math.floor(x0))))
    iy0 = max(0, min(height, int(math.floor(y0))))
    ix1 = max(0, min(width, int(math.ceil(x1))))
    iy1 = max(0, min(height, int(math.ceil(y1))))
    if ix1 - ix0 < 3 or iy1 - iy0 < 3:
        return None
    return rgb[iy0:iy1, ix0:ix1]


def _landmark_xy(
    landmarks: Iterable[Any],
    crop_width: int,
    crop_height: int,
    origin_xy: tuple[int, int],
) -> np.ndarray:
    origin_x, origin_y = origin_xy
    return np.asarray(
        [
            (
                float(point.x) * crop_width + origin_x,
                float(point.y) * crop_height + origin_y,
            )
            for point in landmarks
        ],
        dtype=np.float64,
    )


def _detection_from_landmarks(
    full_rgb: np.ndarray,
    landmarks: Iterable[Any],
    *,
    crop_shape: tuple[int, int],
    origin_xy: tuple[int, int],
    min_face_width_px: float,
    min_mouth_width_px: float,
) -> FaceDetection | None:
    """Convert one MediaPipe face mesh into geometry and image metrics."""
    full_height, full_width = full_rgb.shape[:2]
    crop_height, crop_width = crop_shape
    points = _landmark_xy(landmarks, crop_width, crop_height, origin_xy)
    if points.ndim != 2 or points.shape[0] <= max(_LIP_SHAPE_INDICES) or points.shape[1] != 2:
        return None
    if not np.all(np.isfinite(points)):
        return None

    raw_x0 = float(np.min(points[:, 0]))
    raw_y0 = float(np.min(points[:, 1]))
    raw_x1 = float(np.max(points[:, 0]))
    raw_y1 = float(np.max(points[:, 1]))
    face_width_px = raw_x1 - raw_x0
    face_height_px = raw_y1 - raw_y0
    if face_width_px <= 1.0 or face_height_px <= 1.0:
        return None

    left_corner = points[_MOUTH_CORNERS[0]]
    right_corner = points[_MOUTH_CORNERS[1]]
    mouth_axis = right_corner - left_corner
    mouth_width_px = float(np.linalg.norm(mouth_axis))
    if mouth_width_px <= 1e-6:
        return None
    axis_x = mouth_axis / mouth_width_px
    axis_y = np.asarray((-axis_x[1], axis_x[0]), dtype=np.float64)

    apertures = np.asarray(
        [
            abs(float(np.dot(points[lower] - points[upper], axis_y)))
            for upper, lower in _INNER_LIP_PAIRS
        ],
        dtype=np.float64,
    )
    mouth_open = float(np.average(apertures, weights=_INNER_LIP_WEIGHTS) / mouth_width_px)
    central_lip_apertures = np.asarray(
        [
            abs(float(np.dot(points[lower] - points[upper], axis_y)))
            / mouth_width_px
            for upper, lower in _CENTRAL_LIP_PAIRS
        ],
        dtype=np.float64,
    )
    lip_aperture_spread = float(np.ptp(central_lip_apertures))

    mouth_midpoint = (left_corner + right_corner) * 0.5
    shape: list[float] = []
    for index in _LIP_SHAPE_INDICES:
        relative = points[index] - mouth_midpoint
        shape.extend(
            (
                float(np.dot(relative, axis_x) / mouth_width_px),
                float(np.dot(relative, axis_y) / mouth_width_px),
            )
        )

    lip_points = points[np.asarray(_LIP_SHAPE_INDICES)]
    mouth_x0, mouth_y0 = np.min(lip_points, axis=0)
    mouth_x1, mouth_y1 = np.max(lip_points, axis=0)
    mouth_margin_x = max(2.0, (mouth_x1 - mouth_x0) * 0.18)
    mouth_margin_y = max(2.0, (mouth_y1 - mouth_y0) * 0.55)
    mouth_patch = _extract_patch(
        full_rgb,
        (
            mouth_x0 - mouth_margin_x,
            mouth_y0 - mouth_margin_y,
            mouth_x1 + mouth_margin_x,
            mouth_y1 + mouth_margin_y,
        ),
    )

    # Inset the face patch slightly so tile borders, name labels, and the
    # background do not masquerade as facial detail.
    inset_x = face_width_px * 0.08
    inset_y = face_height_px * 0.08
    face_patch = _extract_patch(
        full_rgb,
        (raw_x0 + inset_x, raw_y0 + inset_y, raw_x1 - inset_x, raw_y1 - inset_y),
    )
    mouth_texture = _gradient_energy(mouth_patch) if mouth_patch is not None else math.nan
    face_texture = _gradient_energy(face_patch) if face_patch is not None else math.nan
    if np.isfinite(mouth_texture) and np.isfinite(face_texture) and face_texture > 1e-6:
        mouth_texture_ratio = float(mouth_texture / face_texture)
    else:
        mouth_texture_ratio = math.nan

    in_frame = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < full_width)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < full_height)
    )
    landmark_visibility = float(np.mean(in_frame))
    lip_visibility = float(np.mean(in_frame[np.asarray(_LIP_SHAPE_INDICES)]))
    resolution_ok = face_width_px >= min_face_width_px and mouth_width_px >= min_mouth_width_px
    if not resolution_ok or lip_visibility < 0.98:
        quality = 0.0
    else:
        size_quality = min(
            1.0,
            face_width_px / (min_face_width_px * 1.5),
            mouth_width_px / (min_mouth_width_px * 1.5),
        )
        mouth_to_face = mouth_width_px / face_width_px
        pose_quality = _clip01((mouth_to_face - 0.10) / 0.13)
        if np.isfinite(face_texture):
            texture_quality = 0.65 + 0.35 * _clip01(face_texture / 5.0)
        else:
            texture_quality = 0.65
        quality = _clip01(size_quality * pose_quality * landmark_visibility * texture_quality)

    bbox = (
        _clip01(raw_x0 / full_width),
        _clip01(raw_y0 / full_height),
        _clip01(raw_x1 / full_width),
        _clip01(raw_y1 / full_height),
    )
    center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
    return FaceDetection(
        bbox=bbox,
        center=center,
        face_width_px=float(face_width_px),
        mouth_width_px=float(mouth_width_px),
        mouth_open=mouth_open,
        mouth_texture_ratio=mouth_texture_ratio,
        quality=quality,
        normalized_lip_shape=tuple(shape),
        appearance_descriptor=_appearance_descriptor(full_rgb, points),
        central_lip_apertures=tuple(float(value) for value in central_lip_apertures),
        lip_aperture_spread=lip_aperture_spread,
    )


def extract_face_detections(
    full_rgb: np.ndarray,
    result: Any,
    *,
    crop_shape: tuple[int, int] | None = None,
    origin_xy: tuple[int, int] = (0, 0),
    min_face_width_px: float = 96.0,
    min_mouth_width_px: float = 20.0,
) -> list[FaceDetection]:
    """Extract detections from a MediaPipe FaceLandmarker result.

    ``bbox`` and ``center`` are normalized to the *full* input frame.  Pixel
    width metrics retain the original frame's scale.  ``crop_shape`` and
    ``origin_xy`` are used when the detector ran on a normalized ROI crop.
    """
    if full_rgb.ndim != 3 or full_rgb.shape[2] != 3:
        raise ValueError("full_rgb must have shape (height, width, 3)")
    if crop_shape is None:
        crop_shape = full_rgb.shape[:2]
    detections: list[FaceDetection] = []
    for landmarks in getattr(result, "face_landmarks", ()):
        detection = _detection_from_landmarks(
            full_rgb,
            landmarks,
            crop_shape=crop_shape,
            origin_xy=origin_xy,
            min_face_width_px=min_face_width_px,
            min_mouth_width_px=min_mouth_width_px,
        )
        if detection is not None:
            detections.append(detection)
    return detections


class FaceTracker:
    """Greedy tracker within one continuous conferencing shot.

    This is deliberately not an identity recognizer. A caller can mark a
    scene cut so a full-screen active-speaker switch always starts a new face
    segment, even when the next face occupies the same coordinates.
    """

    def __init__(
        self,
        *,
        max_gap_s: float = 1.5,
        max_reconnect_gap_s: float = 0.0,
        max_center_distance_boxes: float = 1.15,
        minimum_iou: float = 0.04,
        max_active_appearance_distance: float = 0.68,
        max_reconnect_appearance_distance: float = 0.18,
        reconnect_ambiguity_margin: float = 0.025,
    ) -> None:
        if max_gap_s <= 0:
            raise ValueError("max_gap_s must be positive")
        self.max_gap_s = float(max_gap_s)
        self.max_reconnect_gap_s = float(max_reconnect_gap_s)
        self.max_center_distance_boxes = float(max_center_distance_boxes)
        self.minimum_iou = float(minimum_iou)
        self.max_active_appearance_distance = float(max_active_appearance_distance)
        self.max_reconnect_appearance_distance = float(max_reconnect_appearance_distance)
        self.reconnect_ambiguity_margin = float(reconnect_ambiguity_margin)
        self._tracks: list[FaceTrack] = []
        self._next_track_id = 0
        self._scene_track_floor = 0

    @property
    def tracks(self) -> list[FaceTrack]:
        return sorted(self._tracks, key=lambda track: track.track_id)

    def start_new_scene(self) -> None:
        """Prevent tracks from crossing a detected full-frame scene cut."""
        self._scene_track_floor = self._next_track_id

    @staticmethod
    def _sample(time_s: float, detection: FaceDetection) -> FaceSample:
        return FaceSample(
            time_s=float(time_s),
            mouth_open=detection.mouth_open,
            mouth_texture_ratio=detection.mouth_texture_ratio,
            quality=detection.quality,
            bbox=detection.bbox,
            normalized_lip_shape=detection.normalized_lip_shape,
            mouth_width_px=detection.mouth_width_px,
            central_lip_apertures=detection.central_lip_apertures,
            lip_aperture_spread=detection.lip_aperture_spread,
        )

    def _match_cost(self, track: FaceTrack, detection: FaceDetection) -> float | None:
        if track.last_bbox is None:
            return None
        previous = track.last_bbox
        previous_center = ((previous[0] + previous[2]) * 0.5, (previous[1] + previous[3]) * 0.5)
        center_distance = math.dist(previous_center, detection.center)
        reference_diagonal = max(0.015, (_bbox_diagonal(previous) + _bbox_diagonal(detection.bbox)) * 0.5)
        normalized_distance = center_distance / reference_diagonal
        iou = _bbox_iou(previous, detection.bbox)
        previous_area = max(1e-9, (previous[2] - previous[0]) * (previous[3] - previous[1]))
        current_area = max(1e-9, (detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1]))
        size_change = abs(math.log(current_area / previous_area))
        appearance_distance = _appearance_distance(
            track.appearance_descriptor,
            detection.appearance_descriptor,
        )
        if (
            appearance_distance is not None
            and appearance_distance > self.max_active_appearance_distance
        ):
            return None
        if iou < self.minimum_iou and normalized_distance > self.max_center_distance_boxes:
            return None
        if size_change > math.log(4.0):
            return None
        appearance_cost = 1.35 * appearance_distance if appearance_distance is not None else 0.25
        return normalized_distance + 0.35 * (1.0 - iou) + 0.18 * size_change + appearance_cost

    @staticmethod
    def _update_appearance(track: FaceTrack, detection: FaceDetection) -> None:
        descriptor = detection.appearance_descriptor
        if not descriptor:
            return
        if not track.appearance_descriptor or len(track.appearance_descriptor) != len(descriptor):
            track.appearance_descriptor = descriptor
            track.appearance_sample_count = 1
            return
        # A capped cumulative average adapts slowly to lighting/compression but
        # cannot be overwritten by a single abrupt active-speaker cut.
        effective_count = min(track.appearance_sample_count, 11)
        alpha = 1.0 / (effective_count + 1.0)
        old = np.asarray(track.appearance_descriptor, dtype=np.float64)
        new = np.asarray(descriptor, dtype=np.float64)
        combined = (1.0 - alpha) * old + alpha * new
        norm = float(np.linalg.norm(combined))
        if norm > 1e-12:
            track.appearance_descriptor = tuple(float(value) for value in combined / norm)
            track.appearance_sample_count += 1

    def update(
        self,
        time_s: float,
        detections: Iterable[FaceDetection],
    ) -> list[tuple[int, FaceSample]]:
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        detections = list(detections)
        active_tracks = [
            track
            for track in self._tracks
            if track.track_id >= self._scene_track_floor
            and 0.0 <= time_s - track.last_seen_s <= self.max_gap_s
        ]
        candidates: list[tuple[float, FaceTrack, int]] = []
        for track in active_tracks:
            for detection_index, detection in enumerate(detections):
                cost = self._match_cost(track, detection)
                if cost is not None:
                    candidates.append((cost, track, detection_index))

        # A person may disappear during active-speaker view and return minutes
        # later at the exact same screen location.  Reconnect only on a strict,
        # unambiguous appearance match; otherwise create a new track.
        inactive_tracks = [
            track
            for track in self._tracks
            if self.max_reconnect_gap_s > 0
            and track.track_id >= self._scene_track_floor
            and track not in active_tracks
            and 0.0 <= time_s - track.last_seen_s <= self.max_reconnect_gap_s
            and track.appearance_descriptor
        ]
        for detection_index, detection in enumerate(detections):
            distances = [
                (distance, track)
                for track in inactive_tracks
                if (
                    distance := _appearance_distance(
                        track.appearance_descriptor,
                        detection.appearance_descriptor,
                    )
                )
                is not None
            ]
            distances.sort(key=lambda item: item[0])
            if not distances or distances[0][0] > self.max_reconnect_appearance_distance:
                continue
            if (
                len(distances) > 1
                and distances[1][0] - distances[0][0] < self.reconnect_ambiguity_margin
            ):
                continue
            distance, track = distances[0]
            candidates.append((1.10 + 2.0 * distance, track, detection_index))
        candidates.sort(key=lambda item: item[0])

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        assignments: list[tuple[int, FaceSample]] = []
        for _, track, detection_index in candidates:
            if track.track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            detection = detections[detection_index]
            sample = self._sample(time_s, detection)
            track.samples.append(sample)
            track.last_bbox = detection.bbox
            track.last_seen_s = float(time_s)
            self._update_appearance(track, detection)
            assignments.append((track.track_id, sample))
            assigned_tracks.add(track.track_id)
            assigned_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            sample = self._sample(time_s, detection)
            track = FaceTrack(
                track_id=self._next_track_id,
                samples=[sample],
                last_bbox=detection.bbox,
                last_seen_s=float(time_s),
                appearance_descriptor=detection.appearance_descriptor,
                appearance_sample_count=1 if detection.appearance_descriptor else 0,
            )
            self._next_track_id += 1
            self._tracks.append(track)
            assignments.append((track.track_id, sample))
        return sorted(assignments, key=lambda item: item[0])


def _roi_pixels(
    roi: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if roi is None:
        return 0, 0, width, height
    x0, y0, x1, y1 = roi
    if not all(math.isfinite(value) for value in roi):
        raise ValueError("ROI values must be finite")
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("ROI must be normalized (x0, y0, x1, y1) within 0..1")
    ix0 = max(0, min(width - 1, int(math.floor(x0 * width))))
    iy0 = max(0, min(height - 1, int(math.floor(y0 * height))))
    ix1 = max(ix0 + 1, min(width, int(math.ceil(x1 * width))))
    iy1 = max(iy0 + 1, min(height, int(math.ceil(y1 * height))))
    return ix0, iy0, ix1, iy1


class FaceAnalyzer:
    """One-pass MediaPipe face/mouth analyzer using true video timestamps."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_faces: int = 4,
        roi: tuple[float, float, float, float] | None = None,
        min_face_width_px: float = 96.0,
        min_mouth_width_px: float = 20.0,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        scene_cut_threshold: float = 0.08,
        tracker: FaceTracker | None = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Face Landmarker model not found: {model_path}")
        if max_faces < 1:
            raise ValueError("max_faces must be at least 1")
        self.roi = roi
        self.min_face_width_px = float(min_face_width_px)
        self.min_mouth_width_px = float(min_mouth_width_px)
        self.tracker = tracker or FaceTracker()
        self.scene_cut_threshold = float(scene_cut_threshold)
        self._previous_scene_signature: np.ndarray | None = None
        self.scene_cut_count = 0
        self._last_timestamp_ms: int | None = None
        options = mp.tasks.vision.FaceLandmarkerOptions(
            # Pin the CPU delegate.  On headless macOS sessions the implicit
            # delegate can try to initialize a Metal graph service and abort
            # the entire process instead of raising a Python exception.
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=int(max_faces),
            min_face_detection_confidence=float(min_face_detection_confidence),
            min_face_presence_confidence=float(min_face_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    @property
    def tracks(self) -> list[FaceTrack]:
        return self.tracker.tracks

    @property
    def last_detector_timestamp_ms(self) -> int | None:
        return self._last_timestamp_ms

    def _detector_timestamp(self, time_s: float) -> int:
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        timestamp_ms = max(0, int(round(time_s * 1000.0)))
        if self._last_timestamp_ms is not None:
            timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def process_frame(
        self,
        time_s: float,
        rgb: np.ndarray,
    ) -> list[tuple[int, FaceSample]]:
        """Analyze one RGB frame and update tracks.

        ``time_s`` is retained unmodified in FaceSample.  A separate rounded,
        strictly increasing millisecond clock is maintained only because the
        MediaPipe VIDEO API requires it.
        """
        if self._landmarker is None:
            raise RuntimeError("FaceAnalyzer is closed")
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")
        if rgb.dtype != np.uint8:
            raise ValueError("rgb must use uint8 samples")
        height, width = rgb.shape[:2]
        y_indices = np.linspace(0, height - 1, num=min(90, height), dtype=int)
        x_indices = np.linspace(0, width - 1, num=min(160, width), dtype=int)
        sampled = rgb[np.ix_(y_indices, x_indices)].astype(np.float32)
        signature = (
            0.2126 * sampled[..., 0]
            + 0.7152 * sampled[..., 1]
            + 0.0722 * sampled[..., 2]
        ) / 255.0
        if self._previous_scene_signature is not None:
            change = float(np.mean(np.abs(signature - self._previous_scene_signature)))
            if change >= self.scene_cut_threshold:
                self.tracker.start_new_scene()
                self.scene_cut_count += 1
        self._previous_scene_signature = signature
        x0, y0, x1, y1 = _roi_pixels(self.roi, width, height)
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
        result = self._landmarker.detect_for_video(image, self._detector_timestamp(time_s))
        detections = extract_face_detections(
            rgb,
            result,
            crop_shape=crop.shape[:2],
            origin_xy=(x0, y0),
            min_face_width_px=self.min_face_width_px,
            min_mouth_width_px=self.min_mouth_width_px,
        )
        return self.tracker.update(float(time_s), detections)

    def finish(self) -> list[FaceTrack]:
        return self.tracks

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> FaceAnalyzer:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
