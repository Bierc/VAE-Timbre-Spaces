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

These results conclude **Trilha 0**, establishing a reliable foundation for further experiments on timbre spaces, latent traversal, interpolation, and conditioning analysis.

---

## 8. Next Steps

- Latent interpolations between instruments and pitches
- Quantitative analysis of pitch invariance vs timbre separation
- Comparison with non-conditional VAE
- Improved audio inversion (e.g., neural vocoder)
- Extension toward timbre transfer experiments

