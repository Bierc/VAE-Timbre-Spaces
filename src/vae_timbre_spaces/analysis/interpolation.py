import numpy as np
import torch


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

__all__ = [
    "lerp",
    "interpolate_between_mus",
    "decode_from_z",
    "decode_full_from_mu_with_overlap_add",
]
