# scripts/check_dataset.py
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


def _find_examples_json(root: Path, split: str) -> Path:
    """
    Tries common NSynth folder layouts:
      data/nsynth-<split>.jsonwav/nsynth-<split>/examples.json
    """
    candidates = [
        root / "data" / f"nsynth-{split}.jsonwav" / f"nsynth-{split}" / "examples.json",
        root / "data" / f"nsynth_{split}.jsonwav" / f"nsynth-{split}" / "examples.json",  # just in case
        root / "data" / f"nsynth-{split}" / "examples.json",  # fallback
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find examples.json. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--cache_root", default="data/nsynth_mel_cache")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--n_batches", type=int, default=2)
    ap.add_argument("--max_keys", type=int, default=None, help="Optional cap on number of keys")
    args = ap.parse_args()

    cache_dir = ROOT / args.cache_root / args.split
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

    examples_path = _find_examples_json(ROOT, args.split)
    examples = json.loads(examples_path.read_text(encoding="utf-8"))

    keys = list(examples.keys())
    if args.max_keys is not None:
        keys = keys[: args.max_keys]

    ds = NsynthMelCacheDataset(keys=keys, examples=examples, cache_dir=cache_dir)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[check_dataset] split={args.split}")
    print(f"[check_dataset] examples.json: {examples_path}")
    print(f"[check_dataset] cache_dir: {cache_dir}")
    print(f"[check_dataset] len(keys)={len(keys)} | len(ds)={len(ds)}")

    for bi, batch in enumerate(dl):
        x, pitch, family, key = batch

        print(f"\n[batch {bi}]")
        print("  x:", tuple(x.shape), "| dtype:", x.dtype, "| min/max:", float(x.min()), float(x.max()))
        print("  pitch:", tuple(pitch.shape), "| dtype:", pitch.dtype, "| min/max:", int(pitch.min()), int(pitch.max()))
        print("  family:", tuple(family.shape), "| dtype:", family.dtype, "| min/max:", int(family.min()), int(family.max()))
        print("  key[0:3]:", list(key[:3]))

        # expected x: (B, 1, 80, 128)
        assert x.dim() == 4 and x.shape[1] == 1, "Expected x shape (B,1,80,128)"
        assert x.shape[2] == 80 and x.shape[3] == 128, "Expected mel/time dims (80,128)"
        assert pitch.dim() == 1 and family.dim() == 1, "Expected pitch/family shape (B,)"

        if bi + 1 >= args.n_batches:
            break

    print("\n[check_dataset] OK ✅")


if __name__ == "__main__":
    main()
