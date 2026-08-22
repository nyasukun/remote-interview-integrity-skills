# Third-party notices

## MediaPipe

The toolkit depends on Google's MediaPipe Python package. MediaPipe is
distributed upstream under the Apache License 2.0. This repository does not
redistribute the MediaPipe package, its bundled third-party NOTICE, or model
binaries. Review the notices included with the MediaPipe distribution you
install.

## Face Landmarker task bundle

This public repository does not redistribute the Face Landmarker task bundle.
After reviewing the provider's terms and explicitly approving network access,
users may fetch the official MediaPipe Face Landmarker float16 task from:

<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task>

Expected file SHA-256:

`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`

The helper `scripts/fetch_face_landmarker.py` downloads only this model and
fails unless its SHA-256 matches. The model is then used locally for
lip-landmark measurement; interview media is not sent to the model provider.
Confirm any model-specific terms before downloading or redistributing it.

## FFmpeg libraries through PyAV

PyAV wheels may include FFmpeg libraries and their corresponding license
notices. Encoding availability and applicable license terms depend on the
installed wheel. This skill does not bundle a separate FFmpeg executable.

## Oxford SyncNet

The larger source project used an optional Oxford SyncNet corroboration pass.
Neither its checkpoint nor its adapted network code is bundled in this skill.
