# VAE Timbre Space — Project Checklist

## ✅ Completed

### Dataset & Representation
- [x] Selection of **NSynth (validation subset)** as the base dataset  
- [x] Loading of `examples.json` and corresponding `.wav` files  
- [x] Extraction of **log-mel spectrograms**  
- [x] Clipping log-mel values to `[-80, 0] dB`  
- [x] Normalization to range `[-1, 1]`  
- [x] Padding / cropping to fixed shape `(1, 80, 128)`  
- [x] Verification of input statistics (min / max / mean / std)  

---

### VAE Architecture
- [x] Convolutional encoder (Conv2D with stride = 2)  
- [x] Separate encoder heads for `mu` and `logvar`  
- [x] Correct implementation of the **reparameterization trick**  
- [x] Symmetric convolutional decoder (ConvTranspose2D)  
- [x] Latent dimension defined (`latent_dim = 32`)  
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

---

### Latent Space Diagnostics
- [x] Monitoring of `mu.mean` and `mu.std`  
- [x] Monitoring of `logvar.mean` and `logvar.std`  
- [x] Computation of **raw KL per sample**  
- [x] Pairwise distance computation between `mu` vectors  
- [x] Verification that latent space does not collapse  

---

### Training Pipeline
- [x] Implementation of `NSynthDataset`  
- [x] Train / validation `DataLoader`  
- [x] Epoch-based training loop  
- [x] Logging of relevant metrics (recon, KL raw, KL fb)  
- [x] CPU training with acceptable runtime (~40s per epoch)  
- [x] Model checkpoint saving (`state_dict` + config)  

---

### Qualitative Evaluation
- [x] Visualization of original vs reconstructed spectrograms  
- [x] Interpretation of harmonic smoothing behavior  
- [x] Understanding of reconstruction vs latent continuity trade-off  

---

## ⏭ Planned / Next Steps

### Architecture Experiments
- [ ] Increase latent dimension to 64  
- [ ] Test deeper encoder (4 convolutional levels)  
- [ ] Increase channel capacity (e.g. 32–64–128)  

---

### Training Improvements
- [ ] Train for more epochs (15–20)  
- [ ] Tune `FREE_BITS` (test 0.25 vs 0.5)  
- [ ] Adjust `WARMUP_STEPS` if needed  
- [ ] Add early stopping based on validation reconstruction  

---

### Timbre Space Exploration (Core Goal)
- [ ] Latent interpolation between distinct instruments  
- [ ] Systematic visualization of latent trajectories  
- [ ] Extraction of latent embeddings (`mu`)  
- [ ] PCA / UMAP projection of latent space  
- [ ] Color embeddings by:
  - instrument family  
  - pitch  
  - source  

---

### Musical Evaluation
- [ ] Convert reconstructed log-mel back to audio (optional)  
- [ ] Auditory evaluation of interpolations  
- [ ] Assess perceptual continuity of timbre transitions  

---

### Engineering & Refactoring
- [ ] Refactor notebook into `.py` scripts  
- [ ] Separate modules:
  - dataset  
  - model  
  - training  
  - evaluation  
- [ ] Configuration via JSON / YAML  
- [ ] Add feature caching (precomputed log-mel)  

---

### Writing & Documentation
- [ ] Formal description of the VAE architecture  
- [ ] Justification of beta-VAE and free-bits strategy  
- [ ] Discussion of reconstruction vs timbre trade-off  
- [ ] Connection to IR-CAM / Magenta related work  
