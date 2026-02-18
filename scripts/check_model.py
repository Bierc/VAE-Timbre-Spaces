# scripts/check_model.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure project/src is on path when running from repo root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vae_timbre_spaces.dataset import NsynthMelCacheDataset  # noqa: E402
from vae_timbre_spaces.models import ConditionalVAE  # noqa: E402

import torch
import torch.nn as nn



def _find_examples_json(root: Path, split: str) -> Path:
    candidates = [
        root / "data" / f"nsynth-{split}.jsonwav" / f"nsynth-{split}" / "examples.json",
        root / "data" / f"nsynth_{split}.jsonwav" / f"nsynth-{split}" / "examples.json",
        root / "data" / f"nsynth-{split}" / "examples.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find examples.json. Tried:\n" + "\n".join(str(p) for p in candidates)
    )

def _load_ckpt(path, map_location):
    """
    Supports:
      - raw state_dict (dict[str, Tensor])
      - dict checkpoints containing state_dict/model_state_dict/etc.
      - full nn.Module saved via torch.save(model, ...)
    Returns: (state_dict, meta_dict)
    """
    obj = torch.load(path, map_location=map_location)  # keep weights_only default for now

    meta = {"ckpt_type": type(obj).__name__}

    # 1) If user saved the full model: torch.save(model, path)
    if isinstance(obj, nn.Module):
        meta["format"] = "nn.Module"
        return obj.state_dict(), meta

    # 2) If dict-like checkpoint
    if isinstance(obj, dict):
        meta["format"] = "dict"
        meta["keys"] = list(obj.keys())

        # Common patterns
        candidate_keys = [
            "state_dict",
            "model_state_dict",
            "cvae_state_dict",
            "vae_state_dict",
            "model",
            "net",
            "generator",
        ]

        # 2a) direct candidates
        for k in candidate_keys:
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                # if it looks like a state_dict (tensor values)
                if all(isinstance(v, torch.Tensor) for v in sd.values()):
                    meta["picked_key"] = k
                    return sd, meta

        # 2b) sometimes nested: obj["model"] is nn.Module
        for k in candidate_keys:
            if k in obj and isinstance(obj[k], nn.Module):
                meta["picked_key"] = k
                meta["nested_module"] = True
                return obj[k].state_dict(), meta

        # 2c) dict itself is a state_dict
        if len(obj) > 0 and all(isinstance(v, torch.Tensor) for v in obj.values()):
            meta["format"] = "raw_state_dict"
            return obj, meta

        # 2d) last resort: find any dict value that is a pure tensor dict
        for k, v in obj.items():
            if isinstance(v, dict) and len(v) > 0 and all(isinstance(t, torch.Tensor) for t in v.values()):
                meta["picked_key"] = k
                meta["format"] = "found_tensor_dict"
                return v, meta

        raise ValueError(
            "Checkpoint is a dict but no state_dict-like entry was found.\n"
            f"Keys: {list(obj.keys())[:50]}"
        )

    # 3) unsupported formats
    raise ValueError(f"Unrecognized checkpoint format: type={type(obj)}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--cache_root", default="data/nsynth_mel_cache")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--cond_dim", type=int, default=16)
    ap.add_argument("--pitch_vocab", type=int, default=128)

    ap.add_argument("--device", default=None, help="cuda / cpu / auto (default)")
    ap.add_argument("--max_keys", type=int, default=200, help="Cap dataset size for quick check")
    args = ap.parse_args()

    # Device
    if args.device is None or args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cache_dir = ROOT / args.cache_root / args.split
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

    examples_path = _find_examples_json(ROOT, args.split)
    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    keys = list(examples.keys())[: args.max_keys]

    ds = NsynthMelCacheDataset(keys=keys, examples=examples, cache_dir=cache_dir)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Build model
    model = ConditionalVAE(
        latent_dim=args.latent_dim,
        pitch_vocab=args.pitch_vocab,
        cond_dim=args.cond_dim,
    ).to(device)
    model.eval()

    # Load weights
    state_dict, meta = _load_ckpt(ckpt_path, map_location=device)
    print("[ckpt meta]", meta)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("[check_model] device:", device)
    print("[check_model] split:", args.split)
    print("[check_model] examples.json:", examples_path)
    print("[check_model] cache_dir:", cache_dir)
    print("[check_model] ckpt:", ckpt_path)
    print("[check_model] load_state_dict strict=False")
    print("  missing keys   :", len(missing))
    if missing:
        print("   -", missing[:10], "..." if len(missing) > 10 else "")
    print("  unexpected keys:", len(unexpected))
    if unexpected:
        print("   -", unexpected[:10], "..." if len(unexpected) > 10 else "")

    # Take one batch
    x, pitch, family, key = next(iter(dl))
    x = x.to(device, non_blocking=True)               # (B,1,80,128)
    pitch = pitch.to(device, non_blocking=True).long()  # (B,)

    # Forward
    with torch.no_grad():
        x_hat, mu, logvar, z = model(x, pitch)

    print("\n[forward shapes]")
    print("  x     :", tuple(x.shape), x.dtype)
    print("  pitch :", tuple(pitch.shape), pitch.dtype)
    print("  x_hat :", tuple(x_hat.shape), x_hat.dtype)
    print("  mu    :", tuple(mu.shape), mu.dtype)
    print("  logvar:", tuple(logvar.shape), logvar.dtype)
    print("  z     :", tuple(z.shape), z.dtype)

    # Sanity checks
    assert x.shape == x_hat.shape, "x_hat must match x shape"
    assert mu.shape[0] == x.shape[0] and mu.dim() == 2, "mu should be (B, latent_dim)"
    assert logvar.shape == mu.shape, "logvar should match mu shape"
    assert z.shape == mu.shape, "z should match mu shape"
    assert int(pitch.min()) >= 0 and int(pitch.max()) < args.pitch_vocab, "pitch out of vocab range"

    # Value ranges
    x_min, x_max = float(x.min()), float(x.max())
    xh_min, xh_max = float(x_hat.min()), float(x_hat.max())
    print("\n[value ranges]")
    print(f"  x     min/max: {x_min:.3f} / {x_max:.3f} (expected ~[-1,1])")
    print(f"  x_hat min/max: {xh_min:.3f} / {xh_max:.3f} (raw decoder output, often near [-1,1] but can exceed)")

    # Also check encoder-alone usage: encoder(x) should work
    with torch.no_grad():
        mu2, logvar2 = model.encoder(x)
    print("\n[encoder-only]")
    print("  mu2/logvar2 shapes:", tuple(mu2.shape), tuple(logvar2.shape))
    assert mu2.shape == mu.shape and logvar2.shape == logvar.shape

    print("\n[check_model] OK ✅")


if __name__ == "__main__":
    main()
