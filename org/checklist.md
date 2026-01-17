# VAE Timbre Space — Project Checklist

## ✅ Completed

### Dataset & Representation
- [x] Selection of **NSynth (validation subset)** as the base dataset  
- [x] Loading of `examples.json` and corresponding `.wav` files  
- [x] Extraction of **log-mel spectrograms**  
- [x] Clipping log-mel values to `[-80, 0] dB`  
- [x] Normalization to range `[-1, 1]`  
- [x] Padding / cropping to fixed shape `(1, 80, 128)`  
- [x] Precomputation and caching of log-mel tensors (`.pt`)  
- [x] Verification of input statistics (min / max / mean / std)  

---

### VAE Architecture
- [x] Convolutional encoder (Conv2D with stride = 2)  
- [x] Separate encoder heads for `mu` and `logvar`  
- [x] Correct implementation of the **reparameterization trick**  
- [x] Symmetric convolutional decoder (ConvTranspose2D)  
- [x] Latent dimension defined (`latent_dim = 32`)  
- [x] Conditioning mechanism implemented (pitch-conditioned VAE)  
- [x] Pitch embedding layer and conditioning injection  
- [x] Full verification of tensor shapes across the network  

---

### Loss Function
- [x] Reconstruction loss (MSE)  
- [x] Correct computation of **raw KL divergence**  
- [x] Implementation of **free bits** strategy  
- [x] Beta-VAE formulation  
- [x] Beta annealing (warm-up schedule)  
- [x] Clear separation of:
  - reconstruction loss  
  - raw KL  
  - free-bits KL  
  - total loss  

---

### Issues Identified & Fixed
- [x] Incorrect input scale (absolute dB → normalized)  
- [x] Posterior collapse detected and resolved  
- [x] KL collapse diagnosed correctly  
- [x] Incorrect `.numpy()` usage fixed with `detach()`  
- [x] Latent interpolation evaluated only after VAE stabilization  
- [x] Small batch size understood as a limitation, not root cause  
- [x] CPU bottleneck diagnosed and mitigated (DataLoader tuning)  

---

### Latent Space Diagnostics
- [x] Monitoring of `mu.mean` and `mu.std`  
- [x] Monitoring of `logvar.mean` and `logvar.std`  
- [x] Computation of **raw KL per sample**  
- [x] Pairwise distance computation between `mu` vectors  
- [x] Verification that latent space does not collapse  
- [x] Qualitative inspection of latent continuity  

---

### Training Pipeline
- [x] Implementation of `NSynthDataset` using cached features  
- [x] Train / validation `DataLoader`  
- [x] Epoch-based training loop  
- [x] Logging of relevant metrics (recon, KL raw, KL fb, total)  
- [x] CUDA training enabled and verified  
- [x] Training with batch size ≥ 128  
- [x] Model checkpoint saving (`state_dict` + config)  

---

### Qualitative Evaluation
- [x] Visualization of original vs reconstructed log-mel spectrograms  
- [x] Visualization of reconstruction error (difference map)  
- [x] Interpretation of harmonic smoothing behavior  
- [x] Understanding of reconstruction vs latent regularization trade-off  

---

### Timbre Space Exploration
- [x] Extraction of latent embeddings (`mu`) from validation set  
- [x] PCA projection of latent space  
- [x] UMAP projection of latent space  
- [x] Visualization with Plotly  
- [x] Coloring by:
  - instrument family (with labels)  
  - pitch  
- [x] Qualitative comparison: baseline VAE vs pitch-conditioned VAE  

---

### Audio Reconstruction
- [x] Denormalization from `[-1,1]` to `[-80,0] dB`  
- [x] Inversion of log-mel using Griffin–Lim  
- [x] Reconstruction from:
  - original log-mel  
  - reconstructed log-mel  
- [x] Comparison with original NSynth `.wav`  
- [x] Visual and auditory inspection of waveform differences  

---

## ⏭ Planned / Next Steps

### Architecture Experiments
- [ ] Increase latent dimension to 64  
- [ ] Test deeper encoder (4 convolutional levels)  
- [ ] Increase channel capacity (e.g. 32–64–128)  
- [ ] Test conditioning with instrument family  
- [ ] Multi-conditioning (pitch + family)  

---

### Training Improvements
- [x] Train for more epochs (15–20)  
- [ ] Tune `FREE_BITS` (test 0.25 vs 0.5)  
- [ ] Adjust `WARMUP_STEPS` based on dataset size  
- [ ] Add early stopping based on validation reconstruction  

---

### Timbre Space Analysis (Core Research Goal)
- [ ] Systematic latent interpolation between instruments  
- [ ] Controlled interpolation with fixed pitch  
- [ ] Visualization of latent trajectories  
- [ ] Analysis of cluster overlap between similar families  
- [ ] Investigation of timbre continuity vs discreteness  

---

### Audio Quality Improvements
- [ ] Reconstruct audio with original duration (no fixed `T`)  
- [ ] Improve inversion quality:
  - [ ] Increase Griffin–Lim iterations  
  - [ ] Tune STFT / mel parameters  
  - [ ] Evaluate neural vocoder (HiFi-GAN / MelGAN)  

---

### Engineering & Refactoring
- [ ] Refactor notebook into modular `.py` scripts  
- [ ] Separate modules:
  - dataset  
  - model  
  - training  
  - evaluation  
- [ ] Centralize configuration (YAML / JSON)  
- [ ] Add experiment logging (CSV / WandB / TensorBoard)  

---

### Writing & Documentation
- [ ] Formal description of the Conditional VAE architecture  
- [ ] Justification of pitch conditioning choice  
- [ ] Discussion of beta-VAE and free-bits strategy  
- [ ] Analysis of reconstruction vs timbre regularization  
- [ ] Comparison with IR-CAM / Magenta related work  
- [ ] Draft figures for PCA / UMAP / reconstruction  
- [ ] Write experimental results section  
