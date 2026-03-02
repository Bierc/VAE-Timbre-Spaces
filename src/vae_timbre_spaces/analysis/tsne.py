from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd


def fit_tsne(
    mu_all: np.ndarray,
    perplexity: int = 30,
    # n_iter: int = 1500,
    init: str = "pca",
    metric: str = "euclidean",
    random_state: int = 42,
) -> Tuple[TSNE, np.ndarray]:
    """Fit t-SNE on latent vectors and return fitted TSNE instance and 2D embeddings.

    Note: sklearn's TSNE does not expose a transform() method; returned `TSNE` object
    is the fitted estimator (mainly for meta access).
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, init=init, metric=metric, random_state=random_state)
    emb2d = tsne.fit_transform(mu_all)
    return tsne, emb2d


def save_tsne_embeddings(path: Path, emb2d: np.ndarray, meta: Dict[str, Any]) -> None:
    """Save embeddings and metadata to a compressed NPZ file.

    Meta values will be converted to arrays or strings as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare meta for saving: convert lists/arrays to numpy arrays, scalars to arrays, others to str
    meta_to_save: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (list, tuple, np.ndarray)):
            meta_to_save[k] = np.array(v)
        elif isinstance(v, (str, int, float, bool)):
            meta_to_save[k] = np.array([v])
        else:
            meta_to_save[k] = np.array([str(v)])

    np.savez_compressed(path, emb2d=emb2d, **meta_to_save)


def load_tsne_embeddings(path: Path) -> Dict[str, Any]:
    """Load TSNE embeddings saved with save_tsne_embeddings.

    Returns dict with keys 'emb2d' and 'meta' (a dict of saved meta entries).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TSNE embeddings file not found: {path}")

    data = np.load(path, allow_pickle=True)
    emb2d = data["emb2d"]

    meta: Dict[str, Any] = {}
    for k in data.files:
        if k == "emb2d":
            continue
        arr = data[k]
        # convert single-element arrays back to scalar if appropriate
        if arr.shape == (1,):
            meta[k] = arr[0].item()
        else:
            meta[k] = arr
    return {"emb2d": emb2d, "meta": meta}


def plot_tsne(
    emb2d: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    save_path: Optional[Path] = None,
    filter_labels: Optional[List[str]] = None,
):
    """Plot t-SNE embeddings using Plotly and optionally save an HTML file.

    If filter_labels is provided, only points whose label is in the list will be plotted.
    """
    emb2d = np.asarray(emb2d)
    labels = np.asarray(labels)

    if filter_labels is not None:
        mask = np.isin(labels, filter_labels)
        plot_x = emb2d[mask, 0]
        plot_y = emb2d[mask, 1]
        plot_labels = labels[mask]
    else:
        plot_x = emb2d[:, 0]
        plot_y = emb2d[:, 1]
        plot_labels = labels

    df = pd.DataFrame({"x": plot_x, "y": plot_y, "label": plot_labels})

    fig = px.scatter(df, x="x", y="y", color="label", title=title, opacity=0.7)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_html(str(save_path))
        except Exception:
            # write_html should normally work; if not, ignore saving but return fig
            pass

    fig.show()
    return fig


__all__ = [
    "fit_tsne",
    "save_tsne_embeddings",
    "load_tsne_embeddings",
    "plot_tsne",
]
