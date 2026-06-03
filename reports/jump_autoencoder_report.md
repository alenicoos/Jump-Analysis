# JumpAutoencoder Report

This report explains why the project uses an autoencoder for temporal anomaly detection, which models are involved, why the anomaly detector is not a Transformer, how training is performed, and what results were obtained.

---

## Goal

The `JumpAutoencoder` is used to detect whether a recorded drop-jump movement looks unusual when compared with normal drop-jump sequences.

The task is not to classify one specific injury or one specific technical error. The task is broader:

```text
Does this full movement sequence look like a normal drop jump?
```

This makes the problem naturally suited to anomaly detection. In the current project, there are many examples of normal drop jumps from the mocap dataset, but there is no large labeled dataset of real pathological or clinically abnormal jumps. A supervised classifier would require many labeled examples for each abnormal movement type, which the project does not yet have.

The autoencoder solves this by learning only the structure of normal movement. At inference time, if a new movement cannot be reconstructed well, it is treated as anomalous.

---

## Why Anomaly Detection

A normal drop jump has a recognizable temporal structure:

1. descent from the box;
2. first landing;
3. knee flexion after contact;
4. push-off;
5. second jump/takeoff.

The exact movement varies between athletes, but normal trials share timing, symmetry, posture, and pitch patterns. The goal is to model this distribution of normal sequences.

Anomaly detection is appropriate because:

- the available real abnormal dataset is limited;
- abnormal movement can appear in many forms, not just one label;
- we want the system to flag unfamiliar movement patterns, even if they were not explicitly defined beforehand;
- the mocap dataset provides a strong set of normal examples;
- synthetic anomalies can be used for evaluation, but not as the only source of truth for supervised training.

The autoencoder therefore acts as a normality model: it learns how normal jumps are shaped in time.

---

## Models Used In The Pipeline

The anomaly pipeline uses two models:

### 1. PitchTransformer

The `PitchTransformer` predicts left and right knee pitch frame by frame from frontal 2D keypoints.

Input:

```text
(T, 34)
```

where each frame contains 17 keypoints represented as x coordinates followed by y coordinates.

Output:

```text
(T, 2)
```

where the two channels are:

- left pitch delta;
- right pitch delta.

The pitch signal adds sagittal-plane information that is difficult to infer from raw frontal keypoints alone.

### 2. JumpAutoencoder

The `JumpAutoencoder` receives the keypoints and predicted pitch together.

Input:

```text
(T, 36)
```

where:

- 34 channels are normalized 2D keypoints;
- 2 channels are left/right pitch predicted by `PitchTransformer`.

Output:

```text
(T, 36)
```

The output is a reconstruction of the same sequence. The anomaly score is based on reconstruction error.

---

## Why An Autoencoder

An autoencoder is trained to compress and reconstruct normal sequences.

The model has a bottleneck:

```text
sequence -> encoder -> latent vector -> decoder -> reconstructed sequence
```

If the input is similar to the training distribution, the model can reconstruct it well. If the input contains a movement pattern that is not represented in the normal training data, reconstruction becomes worse.

This gives a direct anomaly score:

```text
anomaly score = reconstruction error
```

This is useful because the model does not need labels such as `valgus`, `shallow landing`, or `asymmetric flexion` during training. It only needs normal examples.

---

## Why LSTM Autoencoder Instead Of Transformer

The project already uses a Transformer for pitch estimation, but the anomaly detector uses an LSTM autoencoder.

This choice is intentional.

### 1. Dataset size

The normal dataset is small:

- 183 mocap athlete sequences;
- augmented normal variants during training.

Transformers usually need more data to generalize well because self-attention has high capacity and can easily overfit small datasets. For anomaly detection, overfitting is especially dangerous: if the model memorizes training sequences, the reconstruction error becomes less meaningful.

An LSTM has a stronger temporal inductive bias for sequential movement and is more data-efficient in this setting.

### 2. The task is reconstruction, not prediction

The `PitchTransformer` predicts a target signal from keypoints. It benefits from attention because it must map temporal keypoint patterns to pitch.

The autoencoder has a different task: represent the whole sequence compactly and reconstruct it. A compact LSTM bottleneck is a natural fit for this because the model is forced to encode the global movement pattern into a latent vector.

### 3. Lower risk of memorization

The selected architecture is deliberately moderate in size. It is expressive enough to learn the structure of a drop jump, but constrained enough to avoid simply copying every frame.

This matters because anomaly detection requires a useful reconstruction gap:

- normal movement should reconstruct well;
- abnormal movement should reconstruct worse.

A large Transformer autoencoder could reconstruct too much, including abnormal sequences, reducing anomaly sensitivity.

### 4. Simpler and faster inference

The LSTM autoencoder is lightweight and easy to run after the video pipeline. It adds little overhead compared with pose estimation and pitch prediction.

---

## Architecture

The model is implemented in `src/jump_analysis/models/jump_autoencoder.py`.

The saved configuration is:

| Component | Value |
|---|---:|
| input_dim | 36 |
| hidden_dim | 64 |
| latent_dim | 32 |
| num_layers | 2 |

The architecture is:

| Part | Description |
|---|---|
| Encoder | Bidirectional LSTM |
| Pooling | Mean pooling over valid frames |
| Bottleneck | Linear projection to latent vector |
| Decoder | Latent vector repeated across time |
| Reconstruction | Unidirectional LSTM + linear output |

The encoder accepts padded batches and uses a mask so padded frames do not affect the pooled latent representation.

---

## Training Data

Training starts from `data/generated/mocap_sequences.npz`, which contains 183 normal drop-jump sequences from the mocap dataset.

For each normal mocap sequence:

1. the 34 keypoint channels are loaded;
2. `PitchTransformer` predicts the 2 pitch channels;
3. keypoints and pitch are concatenated into a 36-dimensional sequence.

The resulting normal sequence format is:

```text
(seq_len, 36)
```

The model normalization statistics are fitted only on the 183 real normal mocap sequences.

---

## Normal Augmentation

Because 183 normal sequences are too few for a neural sequence model, the training script generates augmented normal examples.

Default setup:

| Data source | Count |
|---|---:|
| Real normal mocap sequences | 183 |
| Synthetic normal training variants | 800 |
| Synthetic normal validation variants | 200 |
| Total training sequences | 983 |

The augmentations are applied consistently to keypoints and pitch:

| Augmentation | Purpose |
|---|---|
| Time warp | Simulates slower/faster jumps |
| Scale | Simulates different camera distances |
| Horizontal shift | Simulates imperfect centering |
| Left/right flip | Uses bilateral symmetry and swaps pitch channels |
| Keypoint noise | Simulates YOLO pose noise |
| Small pitch noise | Simulates pitch-estimation variability |

This makes the autoencoder learn a wider distribution of normal jumps instead of memorizing the 183 original athletes.

---

## Training Objective

The model is trained with masked mean squared error.

For each batch:

1. input sequences are normalized;
2. the autoencoder reconstructs them;
3. reconstruction error is computed only on valid frames, ignoring padding;
4. Adam optimization updates the model;
5. validation reconstruction error is checked every 10 epochs.

The best model is selected by validation reconstruction error.

Default training hyperparameters:

| Hyperparameter | Value |
|---|---:|
| epochs | 300 |
| batch_size | 32 |
| learning_rate | 1e-3 |
| optimizer | Adam |
| scheduler | CosineAnnealingLR |
| gradient clipping | 1.0 |
| dropout | 0.2 during training |

The saved inference checkpoint stores dropout as `0.0`, because dropout is not used at inference time.

---

## Anomaly Score

At inference time, the model reconstructs the input sequence and computes a reconstruction error.

The project uses a high-percentile frame-error score rather than only a simple full-sequence mean. This is important because some anomalies may be brief but intense.

Example:

- a short valgus collapse may last only a few frames;
- a mean over the whole sequence could dilute that error;
- a high-percentile score keeps brief peaks visible.

The model then compares the score with a threshold:

```text
is_anomaly = score > anomaly_threshold
```

The threshold is calibrated from normal mocap sequences. In the training script, the default calibration is the 99th percentile of the normal reconstruction scores.

The saved checkpoint currently stores:

```text
anomaly_threshold = 0.11674317494034768
```

Important implementation note: the threshold and the scoring function must always be kept consistent. If the score definition is changed, the autoencoder should be retrained or at least recalibrated so the threshold matches the current scoring method.

---

## Synthetic Anomaly Evaluation

The model is evaluated on synthetic abnormal trials generated from the mocap normal sequences.

The synthetic anomaly set contains:

```text
183 athletes x 7 anomaly types = 1281 anomalous trials
```

The anomaly types are:

| Type | Meaning |
|---|---|
| `knee_valgus` | Knees collapse inward during flexion |
| `asymmetric_flexion` | One side flexes much more than the other |
| `shallow_landing` | Landing is too stiff and shallow |
| `trunk_lateral_lean` | Trunk leans laterally during landing |
| `wide_stance` | Feet/knees are too wide |
| `narrow_stance` | Feet/knees are too narrow |
| `asymmetric_pitch_only` | Pitch is asymmetric while keypoints remain normal |

These anomalies are intentionally moderate-to-severe. They are not meant to perfectly represent all real clinical cases; they are used to test whether the model reacts to clearly abnormal temporal patterns.

---

## Training Results

The training strategy improved generalization by augmenting the normal set.

The main observed effect was:

```text
train/validation gap reduced from about 4x to about 2x
```

This suggests that the model learned a broader normal movement distribution instead of memorizing the 183 original mocap sequences.

On the synthetic anomaly benchmark, the model detected all seven anomaly families in the historical evaluation:

| Anomaly family | Detection |
|---|---:|
| knee valgus | 100% |
| asymmetric flexion | 100% |
| shallow landing | 100% |
| trunk lateral lean | 100% |
| wide stance | 100% |
| narrow stance | 100% |
| asymmetric pitch only | 100% |

These results show that the autoencoder can separate normal mocap-style drop jumps from deliberately distorted movement patterns.

The result should be interpreted carefully: synthetic anomalies are controlled and stronger than many real-world abnormalities. Real clinical validation will require labeled abnormal jumps from real athletes.

---

## Use In The GUI

During the Streamlit analysis workflow:

1. YOLO captures the jump.
2. `PitchTransformer` predicts left/right pitch over time.
3. The 34 keypoint channels and 2 pitch channels are concatenated.
4. `JumpAutoencoder.is_anomaly(...)` returns:
   - whether the movement is anomalous;
   - the anomaly score.
5. `frame_errors_numpy(...)` returns frame-by-frame reconstruction errors.

The GUI uses the frame-level errors to explain which phase of the movement looked most unusual:

- drop from box;
- landing and flexion;
- push-off and rebound.

This makes the autoencoder not only a pass/fail detector, but also a tool for locating where the movement diverged from normal examples.

---

## Limitations

The current autoencoder is a strong prototype, but it has important limits:

1. **Synthetic anomalies are not real clinical labels.**
   The benchmark proves sensitivity to controlled distortions, not clinical validity.

2. **The normal dataset is still small.**
   There are 183 mocap athletes, which is useful but not enough to cover all real-world camera setups, body types, and execution styles.

3. **Pitch depends on the upstream Transformer.**
   If pitch prediction is wrong, the autoencoder input is affected.

4. **Threshold calibration is critical.**
   The saved threshold must match the exact scoring function used at inference time.

5. **The model detects unusual movement, not the cause.**
   A high reconstruction error does not automatically identify a diagnosis. It indicates that the movement pattern differs from the learned normal distribution.

---

## Conclusion

The `JumpAutoencoder` was chosen because the project needs temporal anomaly detection with limited abnormal labels. A reconstruction-based model trained on normal movement is a natural fit: it learns what normal drop-jump sequences look like and flags movements that are difficult to reconstruct.

An LSTM autoencoder was preferred over a Transformer autoencoder because the dataset is small, the task benefits from a compact temporal bottleneck, and a lower-capacity recurrent model reduces the risk of memorizing both normal and abnormal patterns.

The final system combines:

- YOLO keypoints for 2D motion;
- `PitchTransformer` for sagittal-plane pitch dynamics;
- `JumpAutoencoder` for temporal normality scoring.

This gives the application a movement-level anomaly detector that evaluates the whole jump sequence rather than only isolated keyframes.
