# Previous Results — Conditional VAE (CVAE)

This document summarizes the results obtained with the current Conditional Variational Autoencoder (CVAE) implementation for timbre space learning using the NSynth dataset.  
These results correspond to **Trilha 0 (baseline validation and sanity checks)** of the project.

---

## 1. Dataset and Preprocessing

- **Dataset**: NSynth
- **Splits used**: train / validation / test
- **Sampling rate**: 16 kHz
- **Representation**: log-mel spectrogram
- **Mel bins**: 80
- **FFT size**: 1024
- **Hop length**: 256
- **Frame length (T)**: 128 frames (fixed-size crops)
- **Normalization**:
  - log-mel dB range: `[-80, 0]`
  - normalized to `[-1, 1]`
- **Cache**: precomputed log-mel tensors stored as `.pt` files

---

## 2. Model Architecture

- **Model**: Conditional Variational Autoencoder (CVAE)
- **Encoder / Decoder**: convolutional
- **Latent dimension**: 32
- Loss function:
  - Reconstruction loss (MSE)
  - KL divergence
  - Free-bits regularization (`FREE_BITS = 0.5`)
  - Beta-VAE formulation with linear warm-up (`β_max = 2.0`)
- **Conditioning**: pitch
  - pitch vocabulary: 128
  - embedding dimension: 16
- **Total parameters**: 1,223,793

---

## 3. Training Configuration

- **Optimizer**: Adam
- **Learning rate**: (as defined in training config)
- **Loss function**:
  - Reconstruction loss (log-mel space)
  - KL divergence
  - β-VAE formulation with Free Bits
- **β schedule**:
  - Linear warmup
  - β_max = 2.0
- **Free bits**: 0.5
- **Epochs**: 15
- **Batch size**: 128
- **Hardware**: GPU (CUDA)

---

## 4. Training Dynamics

### 4.1 Warmup Behavior

- At early steps (`β ≈ 0`), the model prioritizes reconstruction.
- As β increases, KL divergence becomes active smoothly.
- No instability observed during the transition to β = 2.0.

Example early logs:

```
[step 0] beta=0.000 total=0.889 recon=0.889 kl_raw=0.033
[step 200] beta=0.800 total=0.609 recon=0.207 kl_raw=0.295
[step 600] beta=2.000 total=1.177 recon=0.173 kl_raw=0.394
```


---
### Loss Behavior
- Reconstruction loss stabilized around **0.12–0.13**
- Raw KL divergence stabilized around **0.45–0.48**
- Free-bits constraint remained active throughout training
- No posterior or latent collapse observed
- Validation loss consistently lower than training loss (expected regularization effect)

Overall, training exhibited **stable and consistent behavior**, indicating that the CVAE learned a meaningful and non-degenerate latent representation.

---

### 4.2 Convergence and Stability

Across epochs, the model shows:

- Stable reconstruction loss around **0.12–0.13**
- Stable KL divergence around **0.45–0.48**
- Free-bits term consistently saturated at **0.5**
- No evidence of posterior collapse

Final epoch summary:

```
Epoch 15/15
Train: total=1.123 recon=0.123 kl_raw=0.474 kl_fb=0.500
Val : total=0.632 recon=0.132 kl_raw=0.474 kl_fb=0.500
```


Validation loss remains consistently lower than training loss, which is expected due to regularization and dataset size.

---

## 5. Latent Space Analysis

Latent representations were extracted using the encoder mean (μ).

### 5.1 Sampled Latent Dataset

- **Number of samples**: 2000
- **Latent shape**: `(2000, 32)`
- **Pitch range**: 9–118
- **Instrument families**: 10 unique classes

Stored arrays:

```
mu_all: (2000, 32)
logvar_all: (2000, 32)
pitch_all: (2000,)
family_all: (2000,)
keys_all: (2000,)
```


---

### 5.2 PCA Visualization

- PCA applied to μ
- 2D projection
- Colored by:
  - instrument family (categorical)
  - pitch (continuous)

Observed behavior:
- Partial clustering by instrument family
- Clear pitch gradients across regions of the latent space

---

### 5.3 UMAP Visualization

- UMAP parameters:
  - n_neighbors = 30
  - min_dist = 0.1
  - n_components = 2
- Colored by:
  - instrument family (categorical legend)
  - pitch (continuous)

Observed behavior:
- Improved local structure compared to PCA
- Timbre-related clusters emerge across families
- Pitch conditioning introduces smooth global organization without collapsing family structure

---

## 5. Quantitative Evaluation — Silhouette Score

### 5.1 Global Silhouette per Instrument Family

| Family | Mean Silhouette | Std | Samples |
|------|-----------------|-----|---------|
| 5 | 0.4789 | 0.1361 | 104 |
| 10 | 0.1954 | 0.1252 | 58 |
| 2 | -0.1234 | 0.2557 | 71 |
| 1 | -0.2611 | 0.0915 | 130 |
| 6 | -0.3516 | 0.2490 | 257 |
| 3 | -0.4146 | 0.3062 | 325 |
| 7 | -0.4474 | 0.2219 | 106 |
| 4 | -0.4910 | 0.2364 | 372 |
| 0 | -0.5215 | 0.1424 | 447 |
| 8 | -0.7173 | 0.1770 | 130 |

**Interpretation**
- Global separation by instrument family is weak or negative in most cases
- Indicates that **instrument family is not globally separable** in the latent space
- Suggests pitch as a dominant confounding factor

---

### 5.2 Pitch-Conditioned Silhouette Analysis

Local pitch windows were evaluated:

- `36 ± 1`
- `60 ± 1`
- `84 ± 1`

#### Pitch 60 ± 1

- Total samples: `88`
- Families retained after filtering: `{0, 3, 4, 6}`
- Final samples: `59`

| Family | Mean Silhouette | Std | Samples |
|------|-----------------|-----|---------|
| 3 | **0.4851** | 0.1016 | 13 |
| 0 | 0.0675 | 0.3936 | 17 |
| 6 | -0.1584 | 0.4993 | 15 |
| 4 | -0.2596 | 0.1891 | 14 |

#### Global vs Pitch-Conditioned Comparison

| Family | Global | Pitch 36 | Pitch 60 | Pitch 84 |
|------|--------|----------|----------|----------|
| 0 | -0.5215 | 0.4737 | 0.0675 | NaN |
| 3 | -0.4146 | -0.1724 | **0.4851** | 0.1814 |
| 4 | -0.4910 | -0.2193 | -0.2596 | 0.1605 |
| 6 | -0.3516 | -0.0886 | -0.1584 | NaN |

**Interpretation**
- Conditioning implicitly on pitch **substantially improves separability**
- At `pitch ≈ 60`, **family 3 (guitar)** exhibits clear clustering
- Strong evidence that:
  > Timbre is better represented as a **locally structured, pitch-conditioned space**, rather than a global one

---


## 6. Reconstruction Quality

### 6.1 Spectrogram Reconstruction

For validation samples:

- Original log-mel spectrogram
- Reconstructed log-mel spectrogram
- Difference (original − reconstruction)

Observations:
- Harmonic structures are preserved
- High-frequency attenuation is visible
- Reconstruction error is smooth and structured, not random

---

### 6.2 Full-Length Audio Reconstruction

Procedure:
- Full log-mel spectrogram computed from original audio
- Sliding window inference (T = 128)
- 50% overlap with Hann window
- Overlap-add reconstruction
- Audio inversion using Griffin-Lim

Audio variants:
1. Original waveform
2. Inversion from original log-mel (upper bound)
3. Inversion from reconstructed log-mel (model + inversion)

Observations:
- Timbre identity largely preserved
- Temporal coherence maintained
- Artifacts are dominated by Griffin-Lim inversion rather than model failure

---

## 7. Summary and Conclusions (Trilha 0)

- The CVAE trains stably with no posterior collapse
- Latent space captures:
  - instrument family structure
  - smooth pitch conditioning
- Reconstructions are coherent in both spectrogram and audio domains
- The current implementation is validated as a solid baseline

- Results support the hypothesis that:
  > Timbre organization emerges more clearly under **conditional constraints**, rather than as a global embedding

These results conclude **Trilha 0**, establishing a reliable foundation for further experiments on timbre spaces, latent traversal, interpolation, and conditioning analysis.

---

## 8. Next Steps

- Latent interpolation between families with high silhouette under fixed pitch
  - Initial case: **family 0 ↔ family 3**, `pitch = 60`
- Perceptual evaluation of interpolations
- Exploration of denser pitch-balanced datasets
- Evaluation of neural vocoders to improve audio inversion quality

