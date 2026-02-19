from pathlib import Path
from typing import Union

import numpy as np
import torch
import matplotlib.pyplot as plt

from ..models import ConditionalVAE


def load_model_from_checkpoint(checkpoint_path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> torch.nn.Module:
    """Load ConditionalVAE from a checkpoint saved by save_ckpt or a raw state_dict.

    Compatible formats:
      - dict with keys {"model_state": state_dict, "config": {...}, ...}
      - plain state_dict
    """
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device) if not isinstance(device, torch.device) else device

    ckpt = torch.load(checkpoint_path, map_location=device)

    # determine format
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        config = ckpt.get("config", {}) or {}
    else:
        state_dict = ckpt
        config = {}

    latent_dim = int(config.get("LATENT_DIM", getattr(state_dict, "latent_dim", 32)))
    pitch_vocab = int(config.get("pitch_vocab", 128))
    cond_dim = int(config.get("cond_dim", 16))

    model = ConditionalVAE(latent_dim=latent_dim, pitch_vocab=pitch_vocab, cond_dim=cond_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def reconstruct_one(model: torch.nn.Module, x_norm: Union[np.ndarray, torch.Tensor], pitch: Union[int, torch.Tensor]):
    """Run forward pass and return reconstructed mel in normalized [-1,1].

    x_norm can be numpy or torch, shape (80, T) or (1, 80, T) or (1,1,80,T).
    pitch can be int or torch tensor scalar/1D.
    Returns numpy array shape (80, T).
    """
    # to torch tensor
    if isinstance(x_norm, np.ndarray):
        x_t = torch.tensor(x_norm, dtype=torch.float32)
    else:
        x_t = x_norm.detach().cpu()

    # normalize dims to (1,1,80,T)
    if x_t.ndim == 2:
        x_t = x_t.unsqueeze(0).unsqueeze(0)
    elif x_t.ndim == 3:
        # (1,80,T)
        x_t = x_t.unsqueeze(1)
    elif x_t.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported x_norm ndim={x_t.ndim}")

    x_t = x_t.to(next(model.parameters()).device)

    if isinstance(pitch, torch.Tensor):
        pitch_t = pitch.to(next(model.parameters()).device)
    else:
        pitch_t = torch.tensor([int(pitch)], device=next(model.parameters()).device, dtype=torch.long)

    # If pitch_t is shape (1,), ok. If we have batch>1, user should provide batch aligned.
    # forward
    x_hat, mu, logvar, z = model(x_t, pitch_t)

    # return first element if batch
    xhat_np = x_hat[0, 0].detach().cpu().numpy()
    return xhat_np


def _to_db(x_norm: np.ndarray) -> np.ndarray:
    """Convert normalized [-1,1] -> dB [-80, 0]"""
    x01 = (x_norm + 1.0) / 2.0
    return x01 * 80.0 - 80.0


def plot_mel_reconstruction(x_original_norm: Union[np.ndarray, torch.Tensor], x_recon_norm: Union[np.ndarray, torch.Tensor], title: str = ""):
    """Plot original, reconstruction and absolute difference in dB.

    Accepts inputs in normalized [-1,1]. Supports shapes (80,T), (1,80,T) or (1,1,80,T).
    """
    # convert to numpy
    if isinstance(x_original_norm, torch.Tensor):
        x_o = x_original_norm.detach().cpu().numpy()
    else:
        x_o = np.array(x_original_norm)

    if isinstance(x_recon_norm, torch.Tensor):
        x_r = x_recon_norm.detach().cpu().numpy()
    else:
        x_r = np.array(x_recon_norm)

    # squeeze channel/batch dims
    if x_o.ndim == 4:
        x_o = x_o[0, 0]
    elif x_o.ndim == 3:
        x_o = x_o[0]
    if x_r.ndim == 4:
        x_r = x_r[0, 0]
    elif x_r.ndim == 3:
        x_r = x_r[0]

    # to dB
    x_o_db = _to_db(x_o)
    x_r_db = _to_db(x_r)
    diff_db = np.abs(x_o_db - x_r_db)

    vmin, vmax = -80, 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(x_o_db, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
    axes[0].set_title("Original (dB)")

    im1 = axes[1].imshow(x_r_db, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
    axes[1].set_title("Reconstruction (dB)")

    im2 = axes[2].imshow(diff_db, aspect="auto", origin="lower", cmap="RdBu")
    axes[2].set_title("|Difference| (dB)")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


__all__ = [
    "load_model_from_checkpoint",
    "reconstruct_one",
    "plot_mel_reconstruction",
]
