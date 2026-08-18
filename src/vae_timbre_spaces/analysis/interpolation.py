import numpy as np
import torch

import numpy as _np
from pathlib import Path as _Path
import plotly.express as _px
import plotly.graph_objects as _go

from typing import Optional, Tuple

import pandas as pd


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
        try:
            fig.write_image(str(save_path))
        except Exception:
            print(f"Warning: Failed to save image. Attempting to save interactive HTML instead at {save_path.with_suffix('.html')}")
            try:
                fig.write_html(str(save_path.with_suffix('.html')))
            except Exception:
                pass

    return fig

def plot_pca_with_trajectory(
    emb2d: np.ndarray,
    labels: np.ndarray,
    traj2d: np.ndarray,
    title: str = "",
    save_path: Optional[_Path] = None,
):
    """Plot PCA background scatter colored by labels and overlay trajectory.

    Uses Plotly and returns the figure. If save_path is provided, writes an HTML file.
    """
    df = pd.DataFrame({"x": emb2d[:, 0], "y": emb2d[:, 1], "label": labels})

    fig = _px.scatter(df, x="x", y="y", color="label", title=title, opacity=0.7)

    # trajectory line
    traj_x = traj2d[:, 0]
    traj_y = traj2d[:, 1]

    fig.add_trace(
        _go.Scatter(
            x=traj_x,
            y=traj_y,
            mode="lines+markers",
            line=dict(color="black", width=2),
            marker=dict(size=8, color="black"),
            name="trajectory",
        )
    )

    # start marker (green) and end marker (red)
    fig.add_trace(
        _go.Scatter(x=[traj_x[0]], y=[traj_y[0]], mode="markers", marker=dict(size=12, color="green"), name="start")
    )
    fig.add_trace(
        _go.Scatter(x=[traj_x[-1]], y=[traj_y[-1]], mode="markers", marker=dict(size=12, color="red"), name="end")
    )

    if save_path is not None:
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # prefer image export
            fig.write_image(str(save_path))
        except Exception:
            # fallback to HTML if image export fails
            print(f"Warning: Failed to save image. Attempting to save interactive HTML instead at {save_path.with_suffix('.html')}")
            try:
                fig.write_html(str(save_path.with_suffix('.html')))
            except Exception:
                pass

    return fig


def plot_interpolation_mels(
    x_hat: torch.Tensor,
    alphas: np.ndarray,
    title: str = "",
    n_cols: int = 5,
    save_path: Optional[_Path] = None,
    vmin: float = -80.0,
    vmax: float = 0.0,
    colorscale: str = "Magma",
) -> "plotly.graph_objects.Figure":
    """Plot decoded interpolation mel spectrograms using Plotly.

    Args:
        x_hat: Tensor or array with shape (n_steps, 1, n_mels, T) or (n_steps, C, n_mels, T)
        alphas: array-like of length n_steps
        title: figure title
        n_cols: number of columns in grid
        save_path: optional Path to save HTML (will append .html if missing)
        vmin/vmax: color scale limits in dB
        colorscale: Plotly colorscale name

    Returns:
        plotly.graph_objects.Figure
    """
    from math import ceil
    from plotly.subplots import make_subplots

    # ensure numpy
    if torch.is_tensor(x_hat):
        x_np = x_hat.detach().cpu().numpy()
    else:
        x_np = _np.asarray(x_hat)

    if x_np.ndim != 4:
        raise ValueError(f"x_hat must have shape (n_steps, ch, n_mels, T). Got {x_np.shape}")

    n_steps = int(x_np.shape[0])
    ch = int(x_np.shape[1])

    # pick first channel
    mels = x_np[:, 0, :, :]

    # convert normalized mel to dB using the project's convention
    mels_db = ((mels + 1.0) / 2.0) * 80.0 - 80.0

    n_cols = max(1, int(n_cols))
    n_rows = int(ceil(n_steps / float(n_cols)))

    subplot_titles = [f"α={float(a):.2f}" for a in alphas]
    # pad titles to n_rows*n_cols if needed
    while len(subplot_titles) < n_rows * n_cols:
        subplot_titles.append("")

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles, horizontal_spacing=0.02, vertical_spacing=0.03)

    # add each heatmap as a trace pointing to the shared coloraxis
    for i in range(n_steps):
        r = (i // n_cols) + 1
        c = (i % n_cols) + 1
        z = mels_db[i]
        hm = _go.Heatmap(z=z, zmin=vmin, zmax=vmax, colorscale=colorscale, colorbar=dict(len=0.4), showscale=False)
        fig.add_trace(hm, row=r, col=c)
        # reverse y-axis to emulate origin='lower'
        fig.update_yaxes(autorange='reversed', row=r, col=c)

    # attach a single visible colorbar by adding an invisible heatmap using coloraxis
    # use last subplot for the colorbar placement
    last_r = n_rows
    last_c = ((n_steps - 1) % n_cols) + 1
    cb_trace = _go.Heatmap(z=mels_db[-1], zmin=vmin, zmax=vmax, colorscale=colorscale, colorbar=dict(title='dB', len=0.8), showscale=True)
    fig.add_trace(cb_trace, row=last_r, col=last_c)
    fig.update_yaxes(autorange='reversed', row=last_r, col=last_c)

    fig.update_layout(title_text=title, height=220 * n_rows, width=220 * min(n_cols, n_steps), template='plotly_white')

    if save_path is not None:
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() != '.html':
            save_path = save_path.with_suffix('.html')
        # prefer HTML export (avoids kaleido issues); fall back to image only if HTML fails
        try:
            fig.write_html(str(save_path))
        except Exception:
            try:
                fig.write_image(str(save_path.with_suffix('.png')))
            except Exception:
                pass

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
    "plot_pca_with_trajectory",
    "plot_interpolation_mels",
]
