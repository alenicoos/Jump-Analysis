# Jump Analysis

Project for front-view 2D LESS feature extraction and analysis.

The current feature format has 37 columns:

- 18 front-view features at initial contact (`ic_*`)
- 18 front-view features at maximum knee flexion (`kfmax_*`)
- `crop_length_frames`

## Structure

```text
src/jump_analysis/data/dataset.py        # standard 37-feature column names
src/jump_analysis/data/mocap_dataset.py  # converts the 183-athlete mocap dataset
src/jump_analysis/features/front_2d_features.py # feature math from front-view pose keyframes
src/jump_analysis/models/model.py        # robust reference anomaly detector
src/jump_analysis/video/yolo_video.py    # YOLO webcam/video acquisition
src/jump_analysis/validation/setup_validation.py # setup/camera calibration checks
src/jump_analysis/validation/protocol_validation.py # drop-jump protocol checks
scripts/convert_mocap_dataset.py         # converts Kinematic_Data to 37 features
scripts/main.py                          # main entrypoint for the full workflow
```

## Convert The 183-Athlete Dataset

```bash
python scripts/convert_mocap_dataset.py \
  --root /Users/ale/Kinematic_Data \
  --output mocap_front_37_features.csv
```

For a quick single-subject test:

```bash
python scripts/convert_mocap_dataset.py --subject 909
```

## Compare A Webcam Jump To Mocap

First create the reference CSV:

```bash
python scripts/convert_mocap_dataset.py \
  --root /Users/ale/Kinematic_Data \
  --output mocap_front_37_features.csv
```

Then capture a jump and compare z-scores/percentiles:

```bash
python scripts/main.py
```

If `--height-cm` is not provided, the script asks for the user's height first
and opens the webcam only after the value is entered. The standard reference is
`mocap_front_37_features.csv`, so you do not need to pass `--reference` unless
you want to use another dataset.

The entered height is used in two places:

- setup: estimate the box height from the floor/box calibration;
- normalization: measure the user's body height in pixels during floor setup,
  compute meters-per-pixel from the entered height, then convert shoulder width
  and YOLO keypoints toward mocap-like metric coordinates.

The capture waits while you stand still on the box/chair. Recording starts only
when the ankles drop enough to indicate the beginning of the LESS drop jump.
The head does not need to be visible, but the live capture requires shoulders,
hips, knees, and ankles to stay visible throughout the jump.

Setup validation is intentionally soft for camera geometry: camera roll and
high/low perspective produce warnings, while major floor-vs-box scale changes
and missing box height are treated as errors.
The setup checks should run only after the user is still: `StablePoseBuffer`
collects several frames and returns a median pose only when movement is low.
Warnings and errors can be announced by voice through `AudioFeedback`.

For smoother webcam capture on a laptop, the comparison script defaults to
`yolo26n-pose.pt`. Use `--model yolo26x-pose.pt` only for offline video or if
your machine can run it smoothly.

## Anomaly Detection

`scripts/main.py` now also writes `analysis_result.csv`. The robust
anomaly model treats `mocap_front_37_features.csv` as the normal reference,
computes robust z-scores and central percentile bands, then reports:

- `prediction`: `normal` or `anomaly`
- `anomaly_score`: mean of the worst feature deviations
- `outlier_feature_count`: how many features are outside the normal band
- `worst_feature`: the feature furthest from the reference

All 37 features are still extracted. By default, anomaly detection uses the 36
spatial features and excludes `crop_length_frames`, because frame count depends
on camera FPS and can create false alarms. Use `--include-crop-length` only for
controlled video with comparable frame rates.
