# Drop-Jump Setup, Pose Estimation, Capture, And Protocol Report

This report describes the current AirPose video pipeline: how setup is
performed, which pose measurements are taken, how the jump is tracked, and how
the system decides whether the recorded movement satisfies the drop-jump
protocol.

The active analysis is temporal: YOLO pose frames are captured, protocol
metadata is computed, `PitchTransformer` predicts knee pitch over time, and
`JumpAutoencoder` evaluates the full movement sequence.

---

## Overview

The workflow has three main phases:

1. **Setup and calibration**
   - The user stands on the floor, close to the box.
   - The user then steps onto the box and stays still.
   - The system estimates pixel scale, shoulder width, box height, and setup
     quality.

2. **Drop-jump capture**
   - The user stays still on the box.
   - The system builds an ankle-height baseline.
   - Recording starts automatically when the ankles move downward enough,
     indicating that the drop from the box has started.

3. **Protocol and temporal analysis**
   - The system estimates initial contact and maximum knee flexion.
   - `DropJumpProtocolValidator` checks whether the movement has the minimum
     structure of a valid drop jump.
   - The pose sequence is converted into temporal model input for
     `PitchTransformer` and `JumpAutoencoder`.

---

## Pose Estimation

The video pipeline uses a YOLO pose model with the COCO 17-keypoint layout. The
system does not use a full body mesh; it works from detected 2D keypoints.

The required keypoints are:

| Body part | COCO indices |
|---|---|
| Shoulders | left shoulder 5, right shoulder 6 |
| Hips | left hip 11, right hip 12 |
| Knees | left knee 13, right knee 14 |
| Ankles | left ankle 15, right ankle 16 |

These points are required because the pipeline needs them to estimate body
scale, ankle motion, body center, knee flexion proxy, and the temporal model
input. If any required point is missing, the frame is not accepted as a valid
pose frame.

The nose keypoint is not required during jump capture. It is used only during
setup as one cue for frontal orientation.

During setup, missing required points clear the stable-pose buffer. During jump
capture, missing required points increase the incomplete-frame count. If too few
valid frames are captured, the trial is rejected and the GUI offers a retake.

---

## Person Selection And Tracking

Each YOLO frame can contain zero, one, or multiple people. The system follows
one person across frames:

- if YOLO provides a tracking id, the first valid person becomes the locked
  track;
- if the locked track appears again, that same person is selected;
- if no track id is available yet, the system chooses the person with the
  largest bounding box.

This reduces the risk of switching to another person if someone else enters the
frame.

---

## Setup Phase 1: Floor Pose

The first setup step asks the user to stand on the floor, near the box, facing
the camera. The full body should be visible.

The system waits for a stable pose instead of using a single frame:

- required keypoints must be visible;
- several consecutive frames are collected;
- motion must stay below the `StablePoseBuffer` threshold;
- the final setup pose is the median of the stable frames.

This median pose reduces YOLO jitter and small user movements.

During floor setup the system measures:

| Measurement | How it is obtained | Why it matters |
|---|---|---|
| Floor pose keypoints | Median stable YOLO keypoints | Reference pose for setup |
| Body height in pixels | YOLO person box height | Converts user height into pixel scale |
| Shoulder width in pixels | Distance between shoulders | Estimates real shoulder width |
| Floor ankle height | Mean y-coordinate of both ankles | Reference for box-height detection |

The user-provided height is converted into a pixel scale:

```text
meters_per_pixel = user_height_m / floor_body_height_px
```

This scale is kept because it is still useful for calibration, setup metadata,
and frame-by-frame data collection.

---

## Setup Phase 2: Box Pose

After the floor pose, the user steps onto the box and stays still. The same
stable-pose logic is used again.

The main cue is ankle height. In image coordinates, y increases downward. When
the user steps onto the box, the ankles move upward in the image, so their
y-coordinate becomes smaller.

The estimated box height in pixels is:

```text
box_height_px = floor_ankle_y - box_ankle_y
```

The box pose is accepted only when:

- required body keypoints are visible;
- ankles are clearly higher than in the floor pose;
- the user is stable for enough frames.

The app no longer speaks a separate “box height X centimeters” message, but the
box-height estimate remains in calibration metadata.

---

## Setup Validation

After floor and box poses are captured, `SetupValidator` checks whether the
setup is reliable enough.

There are two result types:

- **warnings**, which inform the user but do not stop the pipeline;
- **errors**, which stop the pipeline because the calibration is unreliable.

### Warning Checks

| Check | What it measures | Purpose |
|---|---|---|
| `camera_roll` | Tilt of shoulders, hips, knees, and ankles | Detects a rotated camera |
| `camera_pitch_or_perspective` | Width changes across body levels | Detects strong perspective distortion |
| `camera_height_perspective` | Vertical segment proportions | Detects camera too high or too low |
| `subject_horizontal_centering` | Body center offset from frame center | Detects subject too far left/right |
| `subject_frontal_orientation` | Nose offset and shoulder asymmetry | Detects if the subject is rotated |

### Error Checks

| Check | What it measures | Why it blocks |
|---|---|---|
| `floor_box_scale_stability` | Change in body scale between floor and box pose | The user moved closer/farther from the camera |
| `box_height_detected` | Ankles are higher on the box than on the floor | The system cannot confirm the user is on a box |

The setup returns a `SetupCalibration` object with:

- floor and box pose keypoints;
- floor body height in pixels;
- meters per pixel;
- measured shoulder width in meters;
- box height in pixels and centimeters;
- scale-change ratio;
- camera roll and perspective estimates.

---

## Drop Preparation And Trigger

After setup, the same camera remains open. The user stays on the box and the
system builds a preparation baseline from the average y-coordinate of the two
ankles:

```text
feet_y = mean(left_ankle_y, right_ankle_y)
```

It also stores body-height estimates from the same preparation frames.

Recording starts when the current ankle position drops below the stable box
baseline by enough pixels:

```text
current_drop_px = feet_y - baseline_feet_y
required_drop_px = min_drop_ratio * median_body_height_px
```

With the default `min_drop_ratio = 0.06`, the ankles must descend by at least
6% of the observed body height.

If the ankles first move upward beyond the allowed tolerance, capture is
rejected. This catches the wrong movement pattern where the user jumps upward
from the box instead of dropping down first.

---

## Captured Frame Data

Every valid captured frame is stored as a `YoloPoseFrame`:

| Field | Meaning |
|---|---|
| `frame_index` | Raw frame counter |
| `keypoints_xy` | Keypoints normalized using measured shoulder width |
| `keypoints_conf` | YOLO confidence for each keypoint |
| `box_xyxy` | YOLO person bounding box |
| `timestamp_s` | Monotonic timestamp |
| `raw_keypoints_xy` | Original pixel keypoints |
| `drop_trigger_px` | Drop measured at trigger time, when available |
| `required_drop_px` | Trigger threshold used at capture time |

Two keypoint representations are kept:

- `raw_keypoints_xy`: pixel coordinates, later normalized by median body height
  for the Transformer and Autoencoder;
- `keypoints_xy`: setup-normalized coordinates, useful for protocol checks and
  scale-aware metadata.

The Transformer input is built as:

```text
(T, 34) = 17 x coordinates + 17 y coordinates
```

using raw pixel keypoints normalized by the median body height across the
recording.

---

## Keyframe Detection

After recording, the system estimates two keyframes:

1. **Initial contact (`ic`)**
   - Estimated from ankle y-coordinate.
   - The system looks for the first frame near the high ankle-y landing level.
   - Since y increases downward, a high ankle y-value means the feet are low in
     the image, which corresponds to landing/contact.

2. **Maximum knee flexion (`kfmax`)**
   - Estimated after initial contact.
   - Combines two cues:
     - knee flexion proxy from the hip-knee-ankle angle;
     - body-center lowering after contact.
   - The final index is the average of the strongest knee-flexion frame and the
     strongest body-lowering frame.

These keyframes are used as protocol anchors and as phase markers in the GUI
explanation.

---

## Protocol Validation

`DropJumpProtocolValidator` decides whether the recorded movement has the
minimum structure of a valid drop jump.

The current checks are:

1. `drop_started_from_height`
2. `second_jump`

The result is stored as metadata:

```text
protocol_passed
drop_started_from_height_passed
drop_started_from_height_value
drop_started_from_height_threshold
second_jump_passed
second_jump_value
second_jump_threshold
```

The protocol passes only if all checks pass.

---

## Protocol Check 1: Drop Started From Height

This verifies that the movement began with a drop from the box.

The preferred measurement is the capture-time trigger:

```text
drop = drop_trigger_px
minimum_required_drop = required_drop_px
```

This is reliable because it comes from the exact moment recording began, while
the user was being compared to the stable box baseline.

If trigger metadata is missing, the validator falls back to the saved sequence:

```text
landing_y = ankle_y at initial contact
start_y = median ankle_y before initial contact
frame_drop = landing_y - start_y
```

The fallback threshold is relative to median shoulder width:

```text
minimum_required_drop = min_drop_height_ratio * reference_width
```

The check passes when:

```text
drop >= minimum_required_drop
```

---

## Protocol Check 2: Second Jump

After landing and maximum knee flexion, a valid drop jump requires a rebound
jump.

The validator searches a window after the larger of:

```text
max_knee_flexion_index
initial_contact_index + 1
```

The default search window is 60 frames.

Inside this window, the validator measures two upward-motion cues:

```text
second_lift = landing_ankle_y - minimum_ankle_y_after_landing
body_lift = body_y_at_landing - minimum_body_y_after_landing
```

Because y increases downward, a smaller y-value means upward movement. The
system uses:

```text
jump_lift = max(second_lift, body_lift)
```

The threshold is:

```text
minimum_second_jump = min_second_jump_ratio * reference_width
```

With the default `min_second_jump_ratio = 0.12`, the second jump must create an
upward movement of at least 12% of median shoulder width.

---

## GUI Failure Handling

If protocol validation fails:

- `protocol_passed` is set to `0`;
- failed checks are stored in metadata;
- Streamlit shows “The movement did not pass the drop-jump protocol checks.”;
- the “Why it failed” table lists observed value, threshold, and explanation;
- the GUI shows a `Retake` button.

If there are too few valid pose frames, the GUI also allows a retake. This
usually means shoulders, hips, knees, or ankles were not consistently visible.

In data collection, an invalid protocol trial is not saved as a valid trial.

---

## Temporal Model Flow

After protocol metadata is computed, the analysis pipeline continues with the
temporal models:

1. `frames_to_transformer_input(frames)` converts valid YOLO frames to `(T, 34)`.
2. `PitchTransformer` predicts `(T, 2)` left/right pitch deltas.
3. The 34 keypoint channels and 2 pitch channels are concatenated into `(T, 36)`.
4. `JumpAutoencoder` reconstructs the sequence and computes an anomaly score.
5. The GUI uses frame-level reconstruction errors to highlight which movement
   phase looked most unusual.

This is the active movement-analysis path used by the app.

---

## Data Collection Output

`collect_jump_data.py` still uses the same setup, capture, and protocol
validation flow. A valid trial folder contains:

- `movement_timeseries.csv`: frame-by-frame timestamps, body scale, keypoints,
  YOLO confidences, bounding box, and aligned IMU columns when available;
- `trial_metadata.json`: setup calibration, protocol metadata, sensor metadata,
  and file list;
- `left_bwt901cl_raw.csv` and `right_bwt901cl_raw.csv` when live sensors are
  used.

---

## Current Scope And Limitations

The protocol validator checks the minimum structure of the movement:

- start from a box/drop height;
- land and perform a second jump.

It does not assign a LESS score and does not fully evaluate clinical quality.
Movement quality is handled separately by the pitch model and temporal
autoencoder.

The protocol result should therefore be interpreted as:

```text
Was this recorded movement structurally a drop jump?
```

not as:

```text
Was this drop jump clinically good?
```
