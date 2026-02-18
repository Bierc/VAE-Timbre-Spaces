import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples


def silhouette_by_family(mu, families):
    sil_samples = silhouette_samples(mu, families, metric="euclidean")

    df = pd.DataFrame({
        "family": families,
        "silhouette": sil_samples
    })

    summary = (
        df.groupby("family")
          .agg(
              mean_silhouette=("silhouette", "mean"),
              std_silhouette=("silhouette", "std"),
              n_samples=("silhouette", "count")
          )
          .sort_values("mean_silhouette", ascending=False)
    )

    return summary


def run_silhouette_for_pitch_window(
    mu_all,
    pitch_all,
    family_all,
    pitch_center,
    tolerance=1,
    min_samples_per_family=10,
):
    mask = np.abs(pitch_all - pitch_center) <= tolerance

    mu_sel = mu_all[mask]
    pitch_sel = pitch_all[mask]
    family_sel = family_all[mask]

    # filtrar famílias com poucos exemplos
    valid_families = [
        f for f in np.unique(family_sel)
        if np.sum(family_sel == f) >= min_samples_per_family
    ]

    fam_mask = np.isin(family_sel, valid_families)

    mu_sel = mu_sel[fam_mask]
    family_sel = family_sel[fam_mask]

    return silhouette_by_family(mu_sel, family_sel)

__all__ = ["silhouette_by_family", "run_silhouette_for_pitch_window"]
