import numpy as np
import torch

import numpy as _np
from pathlib import Path as _Path
import plotly.express as _px
import plotly.graph_objects as _go


def lerp(a, b, alpha):
    return (1 - alpha) * a + alpha * b


def interpolate_between_mus(muA, muB, n_steps=7):
    alphas = np.linspace(0.0, 1.0, n_steps).astype(np.float32)
    mus = [((1 - t) * muA + t * muB) for t in alphas]
    return mus, alphas

@torch.no_grad()
def decode_from_z(cvae, z, pitch):
    cond = cvae.pitch_cond(pitch)
    x_hat = cvae.decoder(z, cond)
    return x_hat

@torch.no_grad()
def decode_full_from_mu_with_overlap_add(
    cvae,
    x_norm_full_ref: np.ndarray,
    mu_vec: torch.Tensor,
    pitch_i: int,
    device,
    win: int = 128,
    hop_win: int = 64,
):
    cvae.eval()
    N_MELS, T_full = x_norm_full_ref.shape

    w = np.hanning(win).astype(np.float32)[None, :]
    y_acc = np.zeros((N_MELS, T_full), dtype=np.float32)
    w_acc = np.zeros((1, T_full), dtype=np.float32)

    z = mu_vec.to(device).unsqueeze(0)
    pitch_t = torch.tensor([pitch_i], device=device, dtype=torch.long)
    cond = cvae.pitch_cond(pitch_t)

    for start in range(0, T_full, hop_win):
        end = start + win
        valid_len = min(win, T_full - start)
        if valid_len <= 0:
            break

        x_hat = cvae.decoder(z, cond)
        chunk_hat = x_hat[0, 0].detach().cpu().numpy()

        y_acc[:, start:start+valid_len] += chunk_hat[:, :valid_len] * w[:, :valid_len]
        w_acc[:, start:start+valid_len] += w[:, :valid_len]

    w_acc = np.maximum(w_acc, 1e-6)
    return y_acc / w_acc


def interpolate_latents(muA: _np.ndarray, muB: _np.ndarray, n_steps: int = 9):
    """Interpolate linearly between two latent vectors.

    Returns:
      traj_mu: ndarray (n_steps, latent_dim)
      alphas: ndarray (n_steps,)
    """
    mus, alphas = interpolate_between_mus(muA, muB, n_steps=n_steps)
    traj_mu = _np.stack([m for m in mus], axis=0)
    return traj_mu, alphas


def select_pair_by_label(labels: _np.ndarray, src_label: str, dst_label: str, seed: int = 0):
    """Deterministically select one index for src_label and one for dst_label.

    labels can be array of strings or ints. Matching is tolerant: if labels are strings we match substring equality or full equality.
    """
    rs = _np.random.RandomState(seed)

    labels_arr = _np.asarray(labels)

    def _indices_for(target):
        # try exact match first
        if labels_arr.dtype.kind in ("U", "S", "O"):
            # string-like labels
            mask = _np.array([str(l) == str(target) for l in labels_arr])
            if mask.sum() == 0:
                # try substring
                mask = _np.array([str(target) in str(l) for l in labels_arr])
        else:
            # numeric labels
            mask = labels_arr == _np.asarray(target, dtype=labels_arr.dtype)
        return _np.where(mask)[0]

    idx_src = _indices_for(src_label)
    idx_dst = _indices_for(dst_label)

    if len(idx_src) == 0:
        raise ValueError(f"No samples found for src_label={src_label}")
    if len(idx_dst) == 0:
        raise ValueError(f"No samples found for dst_label={dst_label}")

    i_src = int(rs.choice(idx_src))
    i_dst = int(rs.choice(idx_dst))
    return i_src, i_dst


def project_trajectory_umap(reducer, traj_mu: _np.ndarray) -> _np.ndarray:
    """Project a trajectory in latent space using an existing UMAP reducer.

    Raises RuntimeError if reducer has no transform method or if transform is not callable.
    """
    if not hasattr(reducer, "transform") or not callable(getattr(reducer, "transform")):
        raise RuntimeError("Provided reducer does not support transform(). Can't project trajectory.")
    return reducer.transform(traj_mu)


def plot_umap_with_trajectory(emb2d: _np.ndarray, labels: _np.ndarray, traj2d: _np.ndarray, title: str = "", save_path: _Path | None = None):
    """Plot background embedding colored by labels and overlay trajectory (2D points).

    Uses Plotly for interactive visualization and optionally saves to HTML.
    """
    df = {
        "x": emb2d[:, 0],
        "y": emb2d[:, 1],
        "label": labels,
    }

    fig = _px.scatter(df, x="x", y="y", color=df["label"], hover_data=["label"], title=title)

    # trajectory line
    fig.add_trace(
        _go.Scatter(
            x=traj2d[:, 0],
            y=traj2d[:, 1],
            mode="lines+markers+text",
            name="trajectory",
            line=dict(color="black", width=2),
            marker=dict(size=8, color="black"),
            text=[f"{i}" for i in range(len(traj2d))],
            textposition="top center",
        )
    )

    # highlight start/end
    fig.add_trace(
        _go.Scatter(
            x=[traj2d[0, 0]], y=[traj2d[0, 1]],
            mode="markers",
            name="start",
            marker=dict(size=12, color="green", symbol="diamond"),
        )
    )
    fig.add_trace(
        _go.Scatter(
            x=[traj2d[-1, 0]], y=[traj2d[-1, 1]],
            mode="markers",
            name="end",
            marker=dict(size=12, color="red", symbol="diamond"),
        )
    )

    if save_path is not None:
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.show()
        fig.write_html(str(save_path))

    return fig


# export new symbols
__all__ = [
    "lerp",
    "interpolate_between_mus",
    "decode_from_z",
    "decode_full_from_mu_with_overlap_add",
    "interpolate_latents",
    "select_pair_by_label",
    "project_trajectory_umap",
    "plot_umap_with_trajectory",
]
