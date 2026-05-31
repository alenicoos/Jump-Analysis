# Jump Analysis

Jump Analysis is a research-oriented Python project for analyzing a drop jump
from a frontal camera view. The goal is to extract pose-based biomechanical
features from a webcam/video recording, compare them with a motion-capture
reference dataset, and use that comparison as a first anomaly-detection layer.

The project is currently focused on a front-view 2D approximation of a LESS-like
workflow. It does not yet produce a clinical LESS score. Instead, it checks that
the recorded movement follows the expected drop-jump protocol and then measures
how far the extracted features are from the reference distribution.

The next required part of the project is sensor supervision. The planned setup
uses two BWT901CL IMU sensors, one near each knee, to record knee orientation
signals during the jump. Those sensor signals can then be treated as ground truth
for learning or correcting knee orientation estimates obtained from video pose
trajectories.

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

## Data Collection Mode

The project also contains a collection workflow for building future supervised
datasets. This mode uses the same height input, floor/box setup, YOLO tracking,
drop detection, and protocol validation as the main workflow, but it does not
run reference comparison or anomaly detection.

Its purpose is to save only clean trials. If setup validation fails or the
drop-jump protocol is not respected, the trial is discarded and no data is
written. If the trial is valid, the project saves:

- the full frame-by-frame pose trajectory in pixels and normalized metric scale;
- video-derived pitch/roll/yaw knee orientation proxies for the whole movement;
- left/right knee sensor pitch/roll/yaw columns aligned to the video timeline
  when BWT901CL CSV exports are provided;
- per-frame body scale data such as body height, shoulder width, hip width, knee
  width, and ankle width;
- the extracted 37 frontal 2D features;
- setup calibration metadata;
- protocol validation metadata.

This keeps raw data collection separate from model evaluation. That separation
will matter once the IMU data are added, because the same trial folder can hold
both the video-derived pose trajectory and the sensor-derived pitch/roll/yaw
signals.

Two supervised model baselines are defined for the sensor stage:

- `VideoOrientationCorrectionModel`: uses video-derived knee pitch/roll/yaw
  proxies plus body-scale/context data to predict the BWT901CL knee orientation
  series;
- `PoseTrajectoryKneeOrientationModel`: uses the full pose trajectory plus
  body-scale/context data, without using the video-derived pitch/roll/yaw proxy,
  to predict the same BWT901CL knee orientation series.

## Repository Structure

```text
scripts/
  main.py
  collect_jump_data.py
  convert_mocap_dataset.py
  list_sensor_ports.py

src/jump_analysis/
  data/
    dataset.py
    mocap_dataset.py
  features/
    front_2d_features.py
    knee_orientation.py
  feedback/
    audio_feedback.py
  models/
    knee_orientation_models.py
    model.py
  sensors/
    bwt901cl_reader.py
    imu_timeseries.py
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

`scripts/collect_jump_data.py` is the validated data collection pipeline. It
uses the same setup and protocol checks as the main workflow, but saves pose
trajectories and feature data only when the jump is valid. It is the intended
entry point for building a future dataset paired with BWT901CL knee IMU signals.

`scripts/convert_mocap_dataset.py` converts the raw 183-athlete motion-capture
dataset into the 37-feature CSV used as reference data.

`scripts/list_sensor_ports.py` lists serial ports visible from Python. It is used
to identify BWT901CL Bluetooth/serial ports on macOS before starting collection.

`data/dataset.py` defines the official 37 feature column names. This file is the
shared contract between feature extraction, dataset conversion, and modeling.

`data/mocap_dataset.py` loads the raw mocap `.mat` files, finds the drop-jump
trial, maps 3D markers to frontal 2D keypoints, and produces rows compatible
with the same feature format used by the webcam pipeline.

`features/front_2d_features.py` contains the geometric feature extraction logic.
It receives already selected 2D keypoints and computes distances, ratios,
frontal knee angles, tilt measures, and body-alignment features.

`features/knee_orientation.py` computes video-derived knee orientation proxies
from the 2D pose trajectory. These are not true 3D joint angles; they are model
inputs to be corrected against IMU ground truth.

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

`models/knee_orientation_models.py` contains the two supervised baselines for
predicting left/right knee pitch, roll, and yaw from either video orientation
proxies or full pose trajectories.

`sensors/bwt901cl_reader.py` reads WITMOTION BWT901CL angle packets from a live
serial/Bluetooth connection and decodes pitch, roll, and yaw.

`sensors/imu_timeseries.py` loads exported IMU orientation CSV files and
interpolates the sensor signals onto the video frame timeline.

`feedback/audio_feedback.py` contains optional voice feedback used during setup.
