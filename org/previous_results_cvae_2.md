# Previous Results — Conditional VAE (CVAE)

This document summarizes the main results obtained with the **Conditional Variational Autoencoder (CVAE)** applied to the **NSynth** dataset, explicitly conditioned on **pitch**. The primary objective is to analyze the organization of the latent space with respect to **instrument families** and to evaluate how pitch conditioning affects timbral separability.

---

## 1. Experimental Setup

### Dataset
- Dataset: **NSynth**
- Splits used:
  - `train`
  - `valid`
  - `test`
- Representation:
  - Log-mel spectrograms
  - 80 mel bins
  - Fixed windows of 128 frames
  - Normalization: `[-80, 0] dB → [-1, 1]`
- Precomputed cache stored as `.pt` files for training efficiency

### Model
- Architecture: **Conditional Variational Autoencoder**
- Latent dimension: `32`
- Conditioning: **pitch (0–127)** via embedding layer
- Convolutional encoder and decoder
- Loss function:
  - Reconstruction loss (MSE)
  - KL divergence
  - Free-bits regularization (`FREE_BITS = 0.5`)
  - Beta-VAE formulation with linear warm-up (`β_max = 2.0`)

---

## 2. Training

- Number of epochs: `15`
- Batch size: `128`
- Optimizer: Adam
- Training split: `train`
- Validation split: `valid`

### Loss Behavior
- Reconstruction loss stabilized around **0.12–0.13**
- Raw KL divergence stabilized around **0.45–0.48**
- Free-bits constraint remained active throughout training
- No posterior or latent collapse observed
- Validation loss consistently lower than training loss (expected regularization effect)

Overall, training exhibited **stable and consistent behavior**, indicating that the CVAE learned a meaningful and non-degenerate latent representation.

---

## 3. Latent Space Extraction

- Number of samples used for analysis: `2000` (validation split)
- Extracted representations: encoder mean vectors `μ`
- Shapes:
  - `mu_all`: `(2000, 32)`
  - `pitch_all`: `(2000,)`
  - `family_all`: `(2000,)`
- Number of instrument families: `10`

---

## 4. Latent Space Visualization

### PCA
- 2D PCA applied to latent means `μ`
- Visualizations:
  - PCA colored by **instrument family**
  - PCA colored by **pitch**

**Observations**
- Strong global overlap between instrument families
- Evidence of local structure emerging under implicit pitch grouping

### UMAP
- 2D UMAP applied to latent means `μ`
- Parameters:
  - `n_neighbors = 30`
  - `min_dist = 0.1`
- Visualizations:
  - UMAP colored by **instrument family**
  - UMAP colored by **pitch**

**Observations**
- Clearer local clustering compared to PCA
- Pitch acts as a dominant organizing factor
- Family separation strongly depends on pitch context

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

## 6. Qualitative Reconstruction Analysis

### Mel-Spectrograms
- Side-by-side visualization of:
  - Original
  - Reconstruction
  - Difference (dB)
- Reconstructions preserve:
  - Global harmonic structure
  - Spectral envelope
- Differences concentrate in:
  - High-frequency regions
  - Low-energy components

### Audio Reconstruction
- Audio inversion performed using **Griffin–Lim**
- Approach:
  - Sliding window + overlap-add
  - Reconstruction with original signal duration
- Observations:
  - Reconstructed audio exhibits metallic artifacts
  - Artifacts are attributed mainly to the **inversion bottleneck**, not to the CVAE itself

---

## 7. Partial Conclusions

- The CVAE learns a **stable and non-collapsed latent space**
- Global instrument-family separation is limited
- **Pitch-conditioned local structure** significantly improves separability
- Results support the hypothesis that:
  > Timbre organization emerges more clearly under **conditional constraints**, rather than as a global embedding

---

## 8. Next Steps

- Latent interpolation between families with high silhouette under fixed pitch
  - Initial case: **family 0 ↔ family 3**, `pitch = 60`
- Perceptual evaluation of interpolations
- Exploration of denser pitch-balanced datasets
- Evaluation of neural vocoders to improve audio inversion quality
