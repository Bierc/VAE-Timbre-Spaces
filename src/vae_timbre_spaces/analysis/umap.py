import umap
import joblib
from pathlib import Path as _Path

def fit_umap(mu_all, n_neighbors=30, min_dist=0.1, random_state=42):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        init="spectral",
        learning_rate=1.0,
        random_state=random_state,
    )
    umap_xy = reducer.fit_transform(mu_all)
    return reducer, umap_xy


def transform_umap(reducer, mu):
    return reducer.transform(mu)


def save_umap_reducer(reducer, path: _Path) -> None:
    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(reducer, str(path))


def load_umap_reducer(path: _Path):
    path = _Path(path)
    if not path.exists():
        raise FileNotFoundError(f"UMAP reducer file not found: {path}")
    return joblib.load(str(path))


__all__ = ["fit_umap", "transform_umap", "save_umap_reducer", "load_umap_reducer"]
