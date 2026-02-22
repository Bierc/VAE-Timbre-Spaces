from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def fit_pca(mu_all: np.ndarray, n_components: int = 2, random_state: int = 42) -> Tuple[PCA, np.ndarray]:
    """Fit PCA on mu_all and return the fitted PCA model and 2D embeddings.

    Parameters
    ----------
    mu_all : np.ndarray
        Array of shape (N, D) with latent means.
    n_components : int
        Number of PCA components to keep (default 2).
    random_state : int
        Random seed forwarded to PCA.

    Returns
    -------
    pca_model : PCA
        Fitted sklearn PCA model.
    emb2d : np.ndarray
        Projected embeddings of shape (N, 2).
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    emb2d = pca.fit_transform(mu_all)
    return pca, emb2d


def transform_pca(pca_model: PCA, mu: np.ndarray) -> np.ndarray:
    """Project latent vectors using a fitted PCA model."""
    return pca_model.transform(mu)


def save_pca_model(pca_model: PCA, path: Path) -> None:
    """Save PCA model to disk using joblib. Create parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca_model, str(path))


def load_pca_model(path: Path) -> PCA:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PCA model not found: {path}")
    return joblib.load(str(path))



__all__ = [
    "fit_pca",
    "transform_pca",
    "save_pca_model",
    "load_pca_model",
]
