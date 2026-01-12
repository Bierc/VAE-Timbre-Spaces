# VAE Timbre Space — Previous Results & Experiment Log

This document records the **model configuration**, **training setup**, and **results obtained so far**.  
It serves as a historical reference for reproducibility and future comparisons.

---

## 1. Dataset Configuration

- Dataset: **NSynth (validation subset)**
- Number of samples:
  - Total: ~12,678
  - Train / Validation split: 90% / 10%
- Audio format:
  - Mono
  - Sample rate: **16,000 Hz**

---

## 2. Input Representation

- Feature: **Log-mel spectrogram**
- Parameters:
  - `n_mels = 80`
  - `n_fft = 1024`
  - `hop_length = 256`
  - Fixed time frames: `T = 128`
- Preprocessing:
  - Power → dB conversion
  - Clipping to `[-80, 0] dB`
  - Normalization to `[-1, 1]`
- Final input shape:
    - (B, 1, 80, 128)

---

## 3. Model Architecture

### Encoder (Convolutional)
- Conv2D layers:
1. `Conv2d(1 → 16, kernel=3, stride=2, padding=1)`
   - Output: `(16, 40, 64)`
2. `Conv2d(16 → 32, kernel=3, stride=2, padding=1)`
   - Output: `(32, 20, 32)`
3. `Conv2d(32 → 64, kernel=3, stride=2, padding=1)`
   - Output: `(64, 10, 16)`
- Flatten dimension:
    -  64 × 10 × 16 = 10240
- Latent heads:
- `fc_mu: Linear(10240 → latent_dim)`
- `fc_logvar: Linear(10240 → latent_dim)`

---

### Latent Space
- Latent dimension:
    - latent_dim = 32

- Sampling:
- Reparameterization trick:
  ```
  z = mu + eps * exp(0.5 * logvar)
  ```

---

### Decoder (Convolutional)
- Linear layer:
- `Linear(latent_dim → 10240)`
- Reshape:
    - (64, 10, 16)

- ConvTranspose2D layers:
1. `ConvTranspose2d(64 → 32, kernel=4, stride=2, padding=1)`
   - Output: `(32, 20, 32)`
2. `ConvTranspose2d(32 → 16, kernel=4, stride=2, padding=1)`
   - Output: `(16, 40, 64)`
3. `ConvTranspose2d(16 → 1, kernel=4, stride=2, padding=1)`
   - Output: `(1, 80, 128)`

---

## 4. Loss Function

### Reconstruction Loss
- Mean Squared Error (MSE)
- Computed on normalized log-mel spectrograms

### KL Divergence
- Raw KL per sample:
    - KL = -0.5 * Σ (1 + logvar − mu² − exp(logvar)) 


### Regularization Strategies
- **Beta-VAE**
- **Free bits**

#### Parameters
- `BETA_MAX = 2.0`
- `WARMUP_STEPS = 2000`
- `FREE_BITS = 0.5`

### Total Loss
- Loss = Recon + beta * KL_free_bits


---

## 5. Training Configuration

- Optimizer: **Adam**
- Learning rate: `1e-3`
- Batch size: `32`
- Device: **CPU**
- Epochs (so far): `5`
- Average epoch time: ~40–45 seconds

---

## 6. Training Dynamics (Observed)

### Beta Annealing
- Beta increases linearly from 0 → 2.0
- As beta increases:
  - Reconstruction loss decreases slowly
  - Total loss increases (expected behavior) (need to understand why)
  - KL remains stable due to free bits

---

## 7. Quantitative Results (Representative)

### Final Epoch (Epoch 5)

**Training**
- Reconstruction loss: ~0.204
- KL raw: ~0.45
- KL (free bits): ~0.50
- Total loss: ~1.01

**Validation**
- Reconstruction loss: ~0.208
- KL raw: ~0.42
- KL (free bits): ~0.50
- Total loss: ~1.10

---

### Latent Statistics (Post-training)
- `mu mean ≈ 0`
- `mu std ≈ 0.20`
- `logvar mean ≈ 0`
- `logvar std ≈ 0.11`

These values indicate **no posterior collapse** and active use of the latent space.

---

### Pairwise Latent Distances (Example Batch)

```
0–1: ~0.30
0–2: ~2.45
0–3: ~0.32
1–2: ~2.42
1–3: ~0.36
2–3: ~2.32
```


This confirms **structured separation** in the latent space.

---

## 8. Qualitative Results

### Reconstruction
- Global spectral envelope correctly reconstructed
- Energy distribution across frequency bands preserved
- Temporal decay captured

### Limitations Observed
- Harmonic structures are smoothed
- Fine spectral lines not fully reconstructed
- Expected behavior due to:
  - log-mel representation
  - latent regularization
  - limited model capacity

---

## 9. Current Interpretation

- The model successfully learns a **continuous, stable timbre representation**
- Reconstruction quality reflects a deliberate trade-off:
  - fidelity ↓
  - latent smoothness ↑
- Suitable foundation for:
  - timbre space exploration
  - interpolation
  - embedding analysis

---

## 10. Status

**Current state:**  
> Stable, non-collapsed VAE with meaningful latent structure, ready for scaling and timbre-space analysis.

