# CVAE Timbre Space — Weekly Action Plan

**Project window:** 08 Feb → 02 Mar  
**Goal:** Finalize implementation, analysis, and article submission

---

## 🗓️ Week 1 — Consolidation & Scale-Up
**08 Feb → 14 Feb**

### Technical Goals
- [x] Finalize NSynth cache for:
  - train
  - validation
  - test
- [x] Unify dataset loading pipeline (cache-based)
- [x] Validate full training loop with CVAE (pitch-conditioned)
- [x] Confirm stable training behavior on full dataset
- [ ] Save final training logs (text + key metrics)
- [ ] Save trained model checkpoint (CVAE)

### Analysis Goals
- [ ] PCA (mu) — color by instrument family
- [ ] PCA (mu) — color by pitch
- [ ] UMAP (mu) — family labels
- [ ] UMAP (mu) — pitch gradient
- [ ] Select **representative plots** for article

### Documentation
- [x] Create `previous_results_cvae.md`
- [ ] Fill sections:
  - Setup Overview
  - Training Behavior
  - Latent Space Analysis (initial)

**Deliverables (end of week):**
- Stable CVAE model
- Final plots selected
- `previous_results_cvae.md` partially filled

---

## 🗓️ Week 2 — Perceptual Validation & Experiments
**15 Feb → 21 Feb**

### Technical Goals
- [ ] Latent interpolation experiments
  - same pitch, different timbres
  - smooth trajectory in latent space
- [ ] Save interpolation examples (mu paths)
- [ ] Improve reconstruction inversion:
  - Increase Griffin-Lim iterations
  - Test overlap size (50% vs 75%)
- [ ] Optional: test higher latent dim (e.g. 64)

### Audio Evaluation
- [ ] Compare:
  - original audio
  - inverted original mel
  - inverted reconstructed mel
- [ ] Identify which artifacts come from:
  - model
  - inversion method

### Documentation
- [ ] Update `previous_results_cvae.md`:
  - Reconstruction
  - Audio inversion
  - Interpolation results
  - Limitations

**Deliverables (end of week):**
- Audio examples ready
- Interpolation plots
- Clear qualitative conclusions

---

## 🗓️ Week 3 — Writing & Finalization
**22 Feb → 02 Mar**

### Writing Goals
- [ ] Abstract
- [ ] Introduction
- [ ] Related Work
- [ ] Methodology
- [ ] Results
- [ ] Discussion
- [ ] Limitations & Future Work
- [ ] Conclusion

### Figures
- [ ] PCA figure
- [ ] UMAP figure
- [ ] Spectrogram comparison
- [ ] Interpolation illustration

### Final Checks
- [ ] Verify reproducibility details
- [ ] Clean figures (labels, captions)
- [ ] Ensure claims match evidence
- [ ] Final proofreading

**Final Deliverables:**
- ✅ Final article
- ✅ Clean repository
- ✅ All results documented and justified

---

## 🔑 Guiding Principle

> **Prefer a well-argued, well-documented model over extra experiments.**  
> This project already has enough depth — clarity is the priority.

---

## 🧠 Notes to Self
- CVAE conditioning on pitch is a *design choice*, not a limitation
- Griffin-Lim artifacts are expected — frame them correctly
- Latent structure > perfect reconstruction
