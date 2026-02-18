import time
import numpy as np
import torch
from typing import Tuple

from .models import ConditionalVAE, vae_loss, beta_schedule


def train_one_epoch(model, loader, optimizer, global_step: int, device, free_bits=0.0, log_every=200):
    model.train()
    stats = {"total": [], "recon": [], "kl_raw": [], "kl_fb": []}

    for x, pitch, family, k in loader:
        x = x.to(device, non_blocking=True)
        pitch = pitch.to(device, non_blocking=True)

        beta = beta_schedule(global_step, warmup_steps=2000, beta_max=1.0)

        x_hat, mu, logvar, z = model(x, pitch)

        total, recon, kl_raw, kl_fb = vae_loss(
            x_hat, x, mu, logvar,
            beta=beta,
            free_bits=free_bits,
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        stats["total"].append(total.item())
        stats["recon"].append(recon.item())
        stats["kl_raw"].append(kl_raw.item())
        stats["kl_fb"].append(kl_fb.item())

        if global_step % log_every == 0:
            print(
                f"[step {global_step}] beta={beta:.3f} "
                f"total={total.item():.3f} recon={recon.item():.3f} "
                f"kl_raw={kl_raw.item():.3f} kl_fb={kl_fb.item():.3f}"
            )

        global_step += 1

    return stats, global_step


@torch.no_grad()
def eval_one_epoch(model, loader, device, free_bits=0.0):
    model.eval()
    stats = {"total": [], "recon": [], "kl_raw": [], "kl_fb": []}

    for x, pitch, family, k in loader:
        x = x.to(device, non_blocking=True)
        pitch = pitch.to(device, non_blocking=True)

        x_hat, mu, logvar, z = model(x, pitch)

        total, recon, kl_raw, kl_fb = vae_loss(
            x_hat, x, mu, logvar,
            beta=1.0,
            free_bits=free_bits,
        )

        stats["total"].append(total.item())
        stats["recon"].append(recon.item())
        stats["kl_raw"].append(kl_raw.item())
        stats["kl_fb"].append(kl_fb.item())

    return stats


def summarize_stats(stats: dict) -> dict:
    return {k: float(np.mean(v)) for k, v in stats.items()}

__all__ = ["train_one_epoch", "eval_one_epoch", "summarize_stats"]

def save_ckpt(path, model, config, global_step):
    state = model.state_dict()
    if len(state) == 0:
        raise RuntimeError("Refusing to save checkpoint: model.state_dict() is empty.")
    torch.save(
        {"model_state": state, "config": config, "global_step": int(global_step)},
        path
    )
