# Jump Analysis

Jump Analysis is a research-oriented Python project for analyzing a drop jump
from a frontal camera view. The goal is to extract pose-based biomechanical
features from a webcam/video recording, compare them with a motion-capture
reference dataset, and use that comparison as a first anomaly-detection layer.

The project is currently focused on a front-view 2D approximation of a LESS-like
workflow. It does not yet produce a clinical LESS score. Instead, it checks that
the recorded movement follows the expected drop-jump protocol and then measures
how far the extracted features are from the reference distribution.

## Approach

The system uses YOLO pose estimation to detect the user and track the relevant
body landmarks. The live workflow starts with a setup phase: the user first
stands on the floor, then on the box. This allows the project to estimate the
body scale from the user's declared height, measure the apparent box height, and
keep the camera calibration consistent during the recording.

After setup, the user starts on the box, drops to the floor, lands, and performs
the rebound jump. The system records the motion only after detecting the drop.
It then verifies three protocol conditions:

- the subject started from a raised position;
- both feet reached contact at approximately the same time;
- a second jump occurred after landing.

If the protocol is valid, the pose sequence is reduced to two keyframes:

- `ic`: initial contact;
- `kfmax`: maximum knee flexion.

For each keyframe, the project extracts 18 frontal 2D features. Together with
the frame distance between the two keyframes, this gives the current 37-feature
representation:

```text
18 ic_* features + 18 kfmax_* features + crop_length_frames = 37 features
```

## Reference Data

The reference dataset comes from the 183-athlete motion-capture dataset. The raw
dataset contains 3D marker data in MATLAB files. The project converts those
markers into a frontal 2D representation compatible with the same feature
extractor used for YOLO pose data.

Webcam data and mocap data are not processed with two
separate feature definitions. Both are converted into the same 37-column format,
so the model can compare them feature by feature.

## Model

The current model is `RobustAnomalyModel`. It treats the converted mocap dataset as the
normal reference distribution and compares a new jump against it.

For every feature, the model estimates robust statistics:

- median;
- median absolute deviation;
- central percentile range.

At prediction time, it computes robust z-scores and counts how many features are
outside the normal reference band. The output is an anomaly-style result:

- `normal` if the jump is close to the reference distribution;
- `anomaly` if too many features, or the strongest feature deviations, are far
  from the reference.

This should be interpreted as a similarity check against the available dataset,
not as a medical diagnosis or a final LESS score.

## Repository Structure

```text
scripts/
  main.py
  convert_mocap_dataset.py

src/jump_analysis/
  data/
    dataset.py
    mocap_dataset.py
  features/
    front_2d_features.py
  feedback/
    audio_feedback.py
  models/
    model.py
  validation/
    setup_validation.py
    protocol_validation.py
  video/
    yolo_video.py
```

`scripts/` contains user-facing entry points. These files are meant to be run
from the terminal and orchestrate the workflow.

`src/jump_analysis/` contains the reusable project code. Keeping the core logic
inside `src` makes the project easier to test, import, and eventually reuse from
a notebook, app, API, or graphical interface.

## Module Roles

`scripts/main.py` is the main runtime pipeline. It asks for the user's height,
opens YOLO and the camera, runs setup validation, captures the drop jump,
extracts the features, validates the protocol, compares the jump to the mocap
reference, and writes the analysis outputs.

`scripts/convert_mocap_dataset.py` converts the raw 183-athlete motion-capture
dataset into the 37-feature CSV used as reference data.

`data/dataset.py` defines the official 37 feature column names. This file is the
shared contract between feature extraction, dataset conversion, and modeling.

`data/mocap_dataset.py` loads the raw mocap `.mat` files, finds the drop-jump
trial, maps 3D markers to frontal 2D keypoints, and produces rows compatible
with the same feature format used by the webcam pipeline.

`features/front_2d_features.py` contains the geometric feature extraction logic.
It receives already selected 2D keypoints and computes distances, ratios,
frontal knee angles, tilt measures, and body-alignment features.

`video/yolo_video.py` handles the camera/video side: frame acquisition, YOLO pose
detection, subject selection, setup capture, keypoint normalization, drop
triggering, frame recording, and conversion from YOLO frames to feature rows.

`validation/setup_validation.py` checks the calibration phase. It compares the
floor pose and box pose, estimates scale from the user's height, measures box
height, and warns about camera geometry issues such as roll or perspective.

`validation/protocol_validation.py` checks whether the recorded movement has the
expected drop-jump structure before the features are interpreted.

`models/model.py` contains the robust anomaly detector used to compare the
captured jump against the reference dataset.

`feedback/audio_feedback.py` contains optional voice feedback used during setup.

