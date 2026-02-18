import torch
import numpy as np
from pathlib import Path

@torch.no_grad()
def extract_mu_logvar(model, loader, n_samples=2000, device=None):
    mus = []
    logvars = []
    pitches = []
    families = []
    keys_out = []

    total = 0
    for x, pitch, family, k in loader:
        x = x.to(device, non_blocking=True)
        pitch_dev = pitch.to(device, non_blocking=True)

        # encoder can be model.encode or model.encoder
        if hasattr(model, "encode") and callable(getattr(model, "encode")):
            out = model.encode(x, pitch_dev) if 'pitch' in model.encode.__code__.co_varnames else model.encode(x)
            mu, logvar = out[0], out[1]
        else:
            try:
                out = model.encoder(x, pitch_dev)
                mu, logvar = out[0], out[1]
            except TypeError:
                out = model.encoder(x)
                mu, logvar = out[0], out[1]

        bsz = mu.shape[0]
        total += bsz

        mus.append(mu.detach().cpu())
        logvars.append(logvar.detach().cpu())
        pitches.append(pitch.detach().cpu())
        families.append(family.detach().cpu())
        keys_out.extend(list(k))

        if total >= n_samples:
            break

    mu_all = torch.cat(mus, dim=0)[:n_samples].numpy()
    logvar_all = torch.cat(logvars, dim=0)[:n_samples].numpy()
    pitch_all = torch.cat(pitches, dim=0)[:n_samples].numpy()
    family_all = torch.cat(families, dim=0)[:n_samples].numpy()
    keys_all = np.array(keys_out[:n_samples])

    return mu_all, logvar_all, pitch_all, family_all, keys_all


def save_embeddings(mu_all, logvar_all, pitch_all, family_all, keys_all, split: str, model_name: str, out_dir: Path = Path("../outputs")):
    """Save embeddings to outputs/embeddings_<split>_<model>.npz and return path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"embeddings_{split}_{model_name}.npz"
    path = out_dir / fname
    np.savez_compressed(
        path,
        mu=mu_all,
        logvar=logvar_all,
        pitch=pitch_all,
        family=family_all,
        keys=keys_all,
    )
    return path


def load_embeddings(npz_path):
    arr = np.load(npz_path)
    return arr

__all__ = ["extract_mu_logvar", "save_embeddings", "load_embeddings"]
