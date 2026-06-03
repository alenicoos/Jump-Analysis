# PitchTransformer Experiments

This report documents the development of the `PitchTransformer`, the temporal model used to estimate left and right knee pitch from a frontal camera video.

The goal was not to infer a full 3D pose from a single image. The goal was narrower and more practical: use the time evolution of YOLO 2D keypoints to estimate how much each knee flexes during a drop jump, frame by frame.

---

## Sensor Setup And Ground Truth

To build a real-world validation set, I recorded 20 personal drop-jump trials while wearing two WITMOTION BWT901CL IMU sensors, one on each leg.

Each sensor was placed on the tibia, approximately halfway between the knee and the ankle. This placement is a practical compromise:

- it is close enough to the knee joint to capture the sagittal-plane motion driven by knee flexion and extension;
- it is far enough from the joint itself to avoid soft-tissue motion and strap movement around the knee;
- it sits on a relatively rigid segment, so the sensor orientation follows the shank more consistently;
- it is easier to mount symmetrically on the left and right legs than a placement very close to the ankle or knee.

The measured IMU pitch is therefore treated as a shank-pitch signal. During a drop jump, shank pitch is strongly related to knee flexion depth: as the knee flexes after landing, the tibia rotates in the sagittal plane. This makes the signal useful as a ground-truth target for training and validating a video-based knee-pitch estimator.

The BWT901CL sensors stream orientation packets over serial/Bluetooth. Each angle packet contains roll, pitch, and yaw as signed values scaled to degrees. The project reads these samples continuously, stores timestamps, and later interpolates the IMU time series onto the video frame timestamps. This gives one left and one right pitch value aligned to every recorded video frame.

For training and validation, pitch is expressed as a delta relative to the trial baseline. In practice, the first stable part of the trial acts as the neutral reference, and the model learns changes in pitch over time rather than absolute sensor orientation. This reduces mounting bias: if a sensor is attached with a small initial tilt, that offset is subtracted out.

---

## Why A Temporal Model

Frontal 2D video does not directly show sagittal-plane knee flexion. The camera sees left-right and vertical motion well, but forward/backward tibia rotation is partially hidden by perspective.

However, the movement is not invisible over time. During landing and knee flexion, the 2D keypoints follow characteristic trajectories:

- knees move relative to ankles and hips;
- body height changes through landing, flexion, and push-off;
- left/right symmetry changes across the sequence;
- the timing between landing, maximum flexion, and takeoff carries useful information.

The `PitchTransformer` uses this temporal context. Instead of predicting pitch from one frame, it receives a sequence of 17 YOLO keypoints over time and predicts left/right pitch for every frame.

---

## Data Sources

### Real IMU Trials

The real validation set contains 20 personal drop-jump trials collected with:

- frontal webcam video;
- YOLO pose keypoints;
- left and right BWT901CL sensors;
- IMU pitch aligned to video timestamps.

These trials are kept as the real-world validation target. They are especially important because they include camera noise, YOLO jitter, sensor mounting imperfections, and real execution variability.

### Mocap Temporal Sequences

The mocap dataset contains 183 athletes performing drop jumps. For the Transformer, the raw `.mat` files are converted into temporal sequences rather than only static keyframes.

For each athlete:

- 3D markers are projected into a frontal 2D view compatible with the YOLO keypoint layout;
- the sequence is extracted around the drop-jump movement;
- the sequence is resampled to 30 fps to match webcam-like timing;
- shank pitch is computed from the 3D knee and ankle markers in the sagittal plane;
- pitch is converted to a delta relative to the initial-contact baseline.

These 183 mocap sequences add cross-subject biomechanical variability that the 20 personal trials alone cannot provide.

### Synthetic Augmentations

The 20 real IMU trials are augmented to produce additional training examples. The augmentations preserve the general movement pattern while increasing robustness:

| Augmentation | Purpose |
|---|---|
| Time warp | Simulates faster or slower executions |
| Scale | Simulates different camera distances |
| Horizontal shift | Simulates imperfect centering |
| Left/right flip | Uses bilateral symmetry and swaps left/right pitch |
| Gaussian keypoint noise | Simulates YOLO estimation noise |

The final Transformer training set combines synthetic variants of the real trials with the mocap temporal sequences.

---

## Model

`PitchTransformer` is a causal Transformer encoder.

Input shape:

```text
(B, T, 34)
```

where each frame contains 17 keypoints, represented as all x coordinates followed by all y coordinates.

Output shape:

```text
(B, T, 2)
```

where the two outputs are:

- left pitch delta in degrees;
- right pitch delta in degrees.

The model uses a causal attention mask, so each frame can only attend to itself and previous frames. This keeps the model compatible with a future streaming setup.

---

## Experiments

### Transformer v1

The first Transformer was trained only on synthetic variants of the 20 real IMU trials.

Configuration:

| Parameter | Value |
|---|---|
| d_model | 64 |
| nhead | 4 |
| num_layers | 3 |
| dim_feedforward | 128 |
| Parameters | ~50k |
| Epochs | 300 |
| Learning rate | 1e-3 |

Validation on the 20 real IMU trials:

| Side | MAE (deg) | RMSE (deg) | Correlation | Bias (deg) |
|---|---:|---:|---:|---:|
| Left | 5.53 | 7.41 | 0.813 | +0.25 |
| Right | 6.28 | 8.12 | 0.796 | -1.74 |

This confirmed that temporal keypoint trajectories contain enough information to estimate pitch changes from frontal video.

### Transformer v2

The second version added 183 mocap temporal sequences to the training set.

Training data:

- 200 synthetic variants from the real IMU trials;
- 183 mocap athlete sequences;
- the same 20 real IMU trials kept for validation.

Configuration:

| Parameter | Value |
|---|---|
| d_model | 64 |
| nhead | 4 |
| num_layers | 3 |
| dim_feedforward | 128 |
| Epochs | 500 |
| Learning rate | 1e-3 |

Validation results:

| Side | MAE (deg) | RMSE (deg) | Correlation | Bias (deg) |
|---|---:|---:|---:|---:|
| Left | 5.88 | 7.67 | 0.825 | +0.20 |
| Right | 6.25 | 7.85 | 0.807 | -1.51 |

The added mocap data improved correlation and reduced right-side bias, but the small model appeared capacity-limited.

### Transformer v3

The final version increased model capacity while keeping the mocap + synthetic training setup.

Configuration:

| Parameter | Value |
|---|---|
| d_model | 128 |
| nhead | 8 |
| num_layers | 4 |
| dim_feedforward | 256 |
| Parameters | ~300k |
| Epochs | 600 |
| Learning rate | 5e-4 |

Validation results:

| Side | MAE (deg) | RMSE (deg) | Correlation | Bias (deg) |
|---|---:|---:|---:|---:|
| Left | 4.20 | 5.94 | 0.847 | -0.44 |
| Right | 4.31 | 5.66 | 0.864 | -0.39 |

This is the selected model. It reduces the typical error to about 4 degrees, keeps left/right correlations around 0.85, and has near-zero average bias on both sides.

---

## Final Summary

| Model | Training data | Parameters | Left MAE | Left r | Right MAE | Right r |
|---|---|---:|---:|---:|---:|---:|
| Transformer v1 | synthetic real trials | ~50k | 5.53 | 0.813 | 6.28 | 0.796 |
| Transformer v2 | synthetic real trials + mocap | ~50k | 5.88 | 0.825 | 6.25 | 0.807 |
| Transformer v3 | synthetic real trials + mocap | ~300k | 4.20 | 0.847 | 4.31 | 0.864 |

The main result is that frontal video can estimate knee-pitch dynamics when the model is given temporal context. The IMU setup provides a real-world validation target, while mocap sequences and synthetic augmentation provide the variability needed to train a model that generalizes beyond the 20 personal trials.

---

## Future Work

The next useful step is to collect additional IMU-labeled trials from more subjects. The current real validation set comes from one person, so adding more athletes would improve subject variability and make the validation more representative.

When new trials are available:

1. collect them with `collect_jump_data.py`;
2. regenerate synthetic trials with `generate_synthetic_trials.py`;
3. retrain `PitchTransformer` with the same v3 configuration;
4. validate again with `validate_pitch_transformer.py`.
