import umap

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

__all__ = ["fit_umap", "transform_umap"]
