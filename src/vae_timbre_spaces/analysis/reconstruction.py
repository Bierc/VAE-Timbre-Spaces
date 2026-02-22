from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

from ..models import ConditionalVAE
from vae_timbre_spaces.analysis.interpolation import interpolate_between_mus, decode_from_z


# -----------------------------
# NSynth family mappings
# -----------------------------
FAMILY_ID_TO_NAME = {
    0: "bass",
    1: "brass",
    2: "flute",
    3: "guitar",
    4: "keyboard",
    5: "mallet",
    6: "organ",
    7: "reed",
    8: "string",
    9: "synth_lead",
    10: "vocal",
}
FAMILY_NAME_TO_ID = {v: k for k, v in FAMILY_ID_TO_NAME.items()}


def family_to_id(name_or_id: Union[str, int]) -> int:
    if isinstance(name_or_id, str):
        name = name_or_id.strip().lower()
        if name not in FAMILY_NAME_TO_ID:
            raise ValueError(f"Unknown family '{name_or_id}'. Options: {sorted(FAMILY_NAME_TO_ID.keys())}")
        return FAMILY_NAME_TO_ID[name]
    return int(name_or_id)


# -----------------------------
# Selection utilities
# -----------------------------

def select_interpolation_pair(
    family_all: np.ndarray,
    pitch_all: np.ndarray,
    keys_all: np.ndarray,
    src_family: Union[str, int],
    dst_family: Union[str, int],
    use_pitch_filter: bool = True,
    pitch_target: int = 60,
    pitch_tol: int = 1,
    src_key_contains: Optional[str] = None,
    dst_key_contains: Optional[str] = None,
    pick_mode: str = "random",
    seed: int = 42,
) -> Tuple[int, int, int]:
    """Select a source/destination pair and a pitch to interpolate.

    Returns (idx_a, idx_b, pitch_interp)
    """
    src_id = family_to_id(src_family)
    dst_id = family_to_id(dst_family)

    idx_src = np.where(family_all.astype(int) == src_id)[0]
    idx_dst = np.where(family_all.astype(int) == dst_id)[0]

    if use_pitch_filter:
        p = pitch_all.astype(int)
        idx_src = idx_src[np.abs(p[idx_src] - pitch_target) <= pitch_tol]
        idx_dst = idx_dst[np.abs(p[idx_dst] - pitch_target) <= pitch_tol]

    if src_key_contains:
        idx_src = np.array([i for i in idx_src if src_key_contains in str(keys_all[i])], dtype=int)
    if dst_key_contains:
        idx_dst = np.array([i for i in idx_dst if dst_key_contains in str(keys_all[i])], dtype=int)

    if len(idx_src) == 0:
        raise RuntimeError(f"No samples for SRC_FAMILY={src_family} after filters. Try disabling pitch/key filters.")
    if len(idx_dst) == 0:
        raise RuntimeError(f"No samples for DST_FAMILY={dst_family} after filters. Try disabling pitch/key filters.")

    rng = np.random.default_rng(seed)
    if pick_mode == "random":
        idx_a = int(rng.choice(idx_src))
        idx_b = int(rng.choice(idx_dst))
    else:
        idx_a = int(idx_src[0])
        idx_b = int(idx_dst[0])

    # choose pitch from A by convention
    pitch_interp = int(pitch_all[idx_a])
    return idx_a, idx_b, pitch_interp


# -----------------------------
# Interpolate & decode
# -----------------------------

def interpolate_and_decode(
    cvae,
    mu_all: np.ndarray,
    idx_a: int,
    idx_b: int,
    n_steps: int,
    pitch_interp: int,
    device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Interpolate latent means between idx_a and idx_b and decode using the CVAE.

    Returns x_hat tensor (n_steps, 1, 80, 128) and alphas array.
    """
    zA = mu_all[idx_a].astype(np.float32)
    zB = mu_all[idx_b].astype(np.float32)

    mus, alphas = interpolate_between_mus(zA, zB, n_steps=n_steps)
    Z_t = torch.tensor(np.stack(mus, axis=0), device=device, dtype=torch.float32)
    pitch_t = torch.full((n_steps,), pitch_interp, device=device, dtype=torch.long)

    with torch.no_grad():
        x_hat = decode_from_z(cvae, Z_t, pitch_t)

    return x_hat, np.array(alphas)


# -----------------------------
# Mel/audio helpers
# -----------------------------

def norm_to_logmel_db(x_norm: np.ndarray) -> np.ndarray:
    x01 = (x_norm + 1.0) / 2.0
    return x01 * 80.0 - 80.0


def mel_db_to_audio(
    x_db: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop: int = 256,
    win: int = 1024,
    griffinlim_iter: int = 64,
    fallback_iter: int = 32,
) -> np.ndarray:
    mel_power = librosa.db_to_power(x_db, ref=1.0)
    try:
        stft_mag = librosa.feature.inverse.mel_to_stft(M=mel_power, sr=sr, n_fft=n_fft, power=1.0)
        wav = librosa.griffinlim(stft_mag, n_iter=griffinlim_iter, hop_length=hop, win_length=win)
    except Exception:
        wav = librosa.feature.inverse.mel_to_audio(
            M=mel_power, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win, n_iter=fallback_iter, power=1.0
        )
    return wav


# -----------------------------
# Save / list audio files
# -----------------------------

def save_interpolation_audio(
    x_hat: torch.Tensor,
    alphas: np.ndarray,
    out_dir: Path,
    model_name: str,
    split: str,
    src_family: str,
    dst_family: str,
    pitch_interp: int,
    sr: int = 16000,
    n_fft: int = 1024,
    hop: int = 256,
    win: int = 1024,
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    n_steps = x_hat.shape[0]
    for i in range(n_steps):
        xhat_norm = x_hat[i, 0].detach().cpu().numpy()
        xhat_db = norm_to_logmel_db(xhat_norm)
        wav = mel_db_to_audio(xhat_db, sr=sr, n_fft=n_fft, hop=hop, win=win)

        out_path = out_dir / f"interp_{model_name}_{split}_{src_family}_to_{dst_family}_p{pitch_interp}_a{alphas[i]:.2f}_{i}.wav"
        sf.write(str(out_path), wav, sr)
        saved.append(out_path)
    return saved


def list_interpolation_files(out_dir: Path, model_name: str, split: str) -> List[Path]:
    pattern = str(Path(out_dir) / f"interp_{model_name}_{split}_*.wav")
    import glob

    files = sorted(glob.glob(pattern))
    return [Path(f) for f in files]


# -----------------------------
# Reconstruction / plotting (moved from previous module)
# -----------------------------

def load_model_from_checkpoint(checkpoint_path: Union[str, Path], device: Union[str, torch.device] = "cpu") -> torch.nn.Module:
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

    # Helper to get config values accepting upper/lower keys
    def _get_config_value(names, default=None, required=False):
        for n in names:
            if n in config:
                return config[n]
        if required:
            raise RuntimeError(f"Checkpoint config missing required key(s): {names}")
        return default

    # latent_dim MUST be present in the checkpoint config (case-insensitive)
    latent_dim = int(_get_config_value(["LATENT_DIM", "latent_dim"], required=True))

    # pitch_vocab: accept keys or fallback only if config is missing/empty
    if config:
        pitch_vocab = int(_get_config_value(["pitch_vocab", "PITCH_VOCAB"], required=True))
    else:
        pitch_vocab = 128

    # cond_dim: accept keys or fallback only if config is missing/empty
    if config:
        cond_dim = int(_get_config_value(["cond_dim", "COND_DIM"], required=True))
    else:
        cond_dim = 16

    model = ConditionalVAE(latent_dim=latent_dim, pitch_vocab=pitch_vocab, cond_dim=cond_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def reconstruct_one(model: torch.nn.Module, x_norm: Union[np.ndarray, torch.Tensor], pitch: Union[int, torch.Tensor]):
    # to torch tensor
    if isinstance(x_norm, np.ndarray):
        x_t = torch.tensor(x_norm, dtype=torch.float32)
    else:
        x_t = x_norm.detach()

    # normalize dims to (1,1,80,T)
    if x_t.ndim == 2:
        x_t = x_t.unsqueeze(0).unsqueeze(0)
    elif x_t.ndim == 3:
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

    x_hat, mu, logvar, z = model(x_t, pitch_t)
    xhat_np = x_hat[0, 0].detach().cpu().numpy()
    return xhat_np


def plot_mel_reconstruction(x_original_norm: Union[np.ndarray, torch.Tensor], x_recon_norm: Union[np.ndarray, torch.Tensor], title: str = ""):
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
    x_o_db = norm_to_logmel_db(x_o)
    x_r_db = norm_to_logmel_db(x_r)
    diff_db = np.abs(x_o_db - x_r_db)

    vmin, vmax = -80, 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(x_o_db, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
    axes[0].set_title("Original (dB)")

    im1 = axes[1].imshow(x_r_db, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
    axes[1].set_title("Reconstruction (dB)")

    im2 = axes[2].imshow(diff_db, aspect="auto", origin="lower", cmap="viridis")
    axes[2].set_title("|Difference| (dB)")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


__all__ = [
    "FAMILY_ID_TO_NAME",
    "FAMILY_NAME_TO_ID",
    "family_to_id",
    "select_interpolation_pair",
    "interpolate_and_decode",
    "norm_to_logmel_db",
    "mel_db_to_audio",
    "save_interpolation_audio",
    "list_interpolation_files",
    "load_model_from_checkpoint",
    "reconstruct_one",
    "plot_mel_reconstruction",
]
