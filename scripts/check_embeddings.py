#!/usr/bin/env python3
# scripts/check_embeddings.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from vae_timbre_spaces.dataset import NsynthMelCacheDataset
from vae_timbre_spaces.models import ConditionalVAE
from vae_timbre_spaces.analysis.embeddings import extract_mu_logvar


# -----------------------------
# Paths / helpers
# -----------------------------

def project_root() -> Path:
    # scripts/check_embeddings.py -> scripts -> repo root
    return Path(__file__).resolve().parents[1]


def nsynth_examples_path(root: Path, split: str) -> Path:
    # data/nsynth-valid.jsonwav/nsynth-valid/examples.json
    return root / "data" / f"nsynth-{split}.jsonwav" / f"nsynth-{split}" / "examples.json"


def nsynth_cache_dir(root: Path, split: str) -> Path:
    return root / "data" / "nsynth_mel_cache" / split


def _load_ckpt_any(path: Path, map_location: str | torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Supports:
      - state_dict directly (dict of tensors)
      - training bundle dict with keys like 'model_state', 'state_dict', 'model'
    Returns: (state_dict, meta)
    """
    obj = torch.load(path, map_location=map_location)  # keep default weights_only for compatibility
    meta: Dict[str, Any] = {"ckpt_type": type(obj).__name__}

    # case A: bundle dict
    if isinstance(obj, dict):
        meta["keys"] = list(obj.keys())

        # common conventions
        for k in ["model_state", "state_dict", "model", "model_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                meta["format"] = "bundle_dict"
                meta["picked_key"] = k
                meta["config"] = obj.get("config", None)
                meta["epoch"] = obj.get("epoch", None)
                meta["global_step"] = obj.get("global_step", None)
                return sd, meta

        # case B: already looks like a state_dict (values are tensors)
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            meta["format"] = "state_dict"
            return obj, meta

    raise ValueError(f"Unrecognized checkpoint format: {path}")


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _summarize_arrays(mu: np.ndarray, logvar: np.ndarray, pitch: np.ndarray, family: np.ndarray) -> None:
    def mm(a: np.ndarray) -> str:
        return f"{float(a.min()):.3f} / {float(a.max()):.3f}"

    print("\n[embeddings summary]")
    print("  mu     :", mu.shape, "min/max:", mm(mu))
    print("  logvar :", logvar.shape, "min/max:", mm(logvar))
    print("  pitch  :", pitch.shape, "min/max:", int(pitch.min()), int(pitch.max()), "| unique:", len(np.unique(pitch)))
    print("  family :", family.shape, "| unique:", len(np.unique(family)))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-test embedding extraction and save to .npz")
    ap.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint (bundle or state_dict).")
    ap.add_argument("--latent_dim", type=int, required=True)
    ap.add_argument("--cond_dim", type=int, default=16)
    ap.add_argument("--pitch_vocab", type=int, default=128)

    ap.add_argument("--n_samples", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0")
    ap.add_argument("--out", type=str, default="", help="Output npz path. Default: outputs/embeddings_smoke_<split>.npz")
    args = ap.parse_args()

    root = project_root()
    split = args.split
    device = _pick_device(args.device)

    examples_path = nsynth_examples_path(root, split)
    cache_dir = nsynth_cache_dir(root, split)
    ckpt_path = (root / args.ckpt) if not Path(args.ckpt).is_absolute() else Path(args.ckpt)

    if not examples_path.exists():
        raise FileNotFoundError(f"examples.json not found: {examples_path}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache_dir not found: {cache_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    out_path = Path(args.out) if args.out else (root / "outputs" / f"embeddings_smoke_{split}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[check_embeddings] device:", device)
    print("[check_embeddings] split:", split)
    print("[check_embeddings] examples.json:", examples_path)
    print("[check_embeddings] cache_dir:", cache_dir)
    print("[check_embeddings] ckpt:", ckpt_path)
    print("[check_embeddings] latent_dim:", args.latent_dim, "| cond_dim:", args.cond_dim)

    # load examples.json
    with open(examples_path, "r") as f:
        examples = json.load(f)

    keys = list(examples.keys())
    ds = NsynthMelCacheDataset(keys=keys, examples=examples, cache_dir=cache_dir)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # build model + load weights
    model = ConditionalVAE(
        latent_dim=args.latent_dim,
        pitch_vocab=args.pitch_vocab,
        cond_dim=args.cond_dim,
    ).to(device)

    state_dict, meta = _load_ckpt_any(ckpt_path, map_location=device)
    print("[ckpt meta]", meta)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[load_state_dict strict=False]")
    print("  missing keys   :", len(missing))
    print("  unexpected keys:", len(unexpected))

    model.eval()

    # extract
    mu_all, logvar_all, pitch_all, family_all, keys_all = extract_mu_logvar(
        model=model,
        loader=loader,
        n_samples=args.n_samples,
        device=device,
    )

    _summarize_arrays(mu_all, logvar_all, pitch_all, family_all)
    print("  keys   :", keys_all.shape, "| example:", keys_all[0] if len(keys_all) else None)

    # save
    np.savez_compressed(
        out_path,
        mu=mu_all,
        logvar=logvar_all,
        pitch=pitch_all.astype(np.int16),
        family=family_all.astype(np.int16),
        keys=keys_all.astype(str),
        meta=np.array(
            [{
                "split": split,
                "n_samples": int(mu_all.shape[0]),
                "latent_dim": int(args.latent_dim),
                "cond_dim": int(args.cond_dim),
                "pitch_vocab": int(args.pitch_vocab),
                "ckpt": str(ckpt_path),
            }],
            dtype=object
        ),
    )
    print(f"\n[check_embeddings] Saved ✅ -> {out_path}")


if __name__ == "__main__":
    main()
