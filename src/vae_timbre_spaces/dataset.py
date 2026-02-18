from pathlib import Path
from torch.utils.data import Dataset
import torch
import json
from typing import Tuple, Dict, List

class NsynthMelCacheDataset(Dataset):
    """Dataset loading precomputed log-mel tensors from a cache directory.

    Each file is expected to be a .pt with a tensor shape (1, N_MELS, T) normalized to [-1,1].
    Returns: x (Tensor), pitch (long), family (long), key (str)
    """
    def __init__(self, keys, examples, cache_dir: Path):
        self.keys = list(keys)
        self.examples = examples
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        pt_path = self.cache_dir / f"{k}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing cache file: {pt_path}")

        x = torch.load(pt_path, weights_only=True)

        pitch = torch.tensor(int(self.examples[k]["pitch"]), dtype=torch.long)
        family = torch.tensor(int(self.examples[k]["instrument_family"]), dtype=torch.long)

        return x, pitch, family, k


def load_examples_json(root: Path) -> Tuple[List[str], Dict]:
    """Load examples.json from a split root and return (keys, examples dict).

    root: path to the split folder that contains examples.json and audio/.
    """
    json_path = Path(root) / "examples.json"
    if not json_path.exists():
        raise FileNotFoundError(f"examples.json not found: {json_path}")

    with open(json_path, "r") as f:
        examples = json.load(f)

    keys = list(examples.keys())
    return keys, examples


__all__ = ["NsynthMelCacheDataset", "load_examples_json"]
