# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Core VAE pieces
# -----------------------------

class VAEConvEncoder(nn.Module):
    """
    Convolutional encoder for log-mel inputs shaped (B, 1, 80, 128).

    Architecture:
      (B,1,80,128)
        -> Conv2d(1->16, k3,s2,p1)  => (B,16,40,64)
        -> Conv2d(16->32,k3,s2,p1)  => (B,32,20,32)
        -> Conv2d(32->64,k3,s2,p1)  => (B,64,10,16)
        -> flatten => (B, 64*10*16 = 10240)
        -> fc_mu, fc_logvar => (B, latent_dim)
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # fixed for input (80,128) with stride=2 three times
        self.C, self.H, self.W = 64, 10, 16
        self.flatten_dim = self.C * self.H * self.W  # 10240

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class ConvDecoder(nn.Module):
    """
    Convolutional decoder that maps z (B,latent_dim) back to (B,1,80,128).

    Architecture:
      z -> fc -> (B,64*10*16) -> reshape (B,64,10,16)
        -> ConvT(64->32,k4,s2,p1) => (B,32,20,32)
        -> ConvT(32->16,k4,s2,p1) => (B,16,40,64)
        -> ConvT(16->1 ,k4,s2,p1) => (B, 1,80,128)
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()

        self.C, self.H, self.W = 64, 10, 16
        self.flatten_dim = self.C * self.H * self.W  # 10240

        self.fc = nn.Linear(latent_dim, self.flatten_dim)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1,  kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.size(0), self.C, self.H, self.W)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x_hat = self.deconv3(x)  # logits / raw reconstruction
        return x_hat


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    z = mu + std * eps, where std = exp(0.5*logvar), eps ~ N(0, I)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAE(nn.Module):
    """
    Baseline VAE:
      encoder(x) -> (mu, logvar)
      z = reparameterize(mu, logvar)
      decoder(z) -> x_hat
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = VAEConvEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z


# -----------------------------
# Loss helpers
# -----------------------------

def kl_raw_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z|x) || p(z)) for diagonal Gaussians, per sample.
    Returns shape: (B,)
    """
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      total, recon, kl_raw_mean, kl_fb_mean

    - recon: MSE mean over all elements
    - kl_raw_mean: mean over batch (raw KL)
    - kl_fb_mean: mean over batch of clamp(KL_per_sample, min=free_bits)
    - total = recon + beta * kl_fb_mean
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")

    kl_ps = kl_raw_per_sample(mu, logvar)    # (B,)
    kl_raw_mean = kl_ps.mean()

    if free_bits > 0.0:
        kl_fb_mean = torch.clamp(kl_ps, min=free_bits).mean()
    else:
        kl_fb_mean = kl_raw_mean

    total = recon + float(beta) * kl_fb_mean
    return total, recon, kl_raw_mean, kl_fb_mean


# -----------------------------
# Beta schedule helper
# -----------------------------

def linear_beta_schedule(global_step: int, warmup_steps: int, beta_max: float) -> float:
    """
    Linear warmup from 0 to beta_max.
    """
    if warmup_steps <= 0:
        return float(beta_max)
    frac = min(1.0, float(global_step) / float(warmup_steps))
    return float(beta_max) * frac


# -----------------------------
# Conditional VAE (pitch-conditioned decoder)
# -----------------------------

class PitchConditioner(nn.Module):
    """
    Maps MIDI pitch (0..127) to a dense conditioning vector via an embedding.
    """
    def __init__(self, vocab_size: int = 128, cond_dim: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.cond_dim = cond_dim
        self.embed = nn.Embedding(vocab_size, cond_dim)

    def forward(self, pitch: torch.Tensor) -> torch.Tensor:
        """
        pitch: (B,) integer tensor in [0, vocab_size-1]
        returns: (B, cond_dim)
        """
        if pitch.dtype != torch.long:
            pitch = pitch.long()
        return self.embed(pitch)


class ConditionalConvDecoder(nn.Module):
    """
    Pitch-conditioned version of ConvDecoder.

    Only change vs baseline decoder:
      fc takes [z, cond] concatenated:
        (B, latent_dim + cond_dim) -> flatten_dim (10240)
    """
    def __init__(self, latent_dim: int = 32, cond_dim: int = 16):
        super().__init__()

        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.C, self.H, self.W = 64, 10, 16
        self.flatten_dim = self.C * self.H * self.W  # 10240

        self.fc = nn.Linear(latent_dim + cond_dim, self.flatten_dim)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1,  kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        z:    (B, latent_dim)
        cond: (B, cond_dim)
        """
        zc = torch.cat([z, cond], dim=1)  # (B, latent_dim+cond_dim)
        x = self.fc(zc).view(z.size(0), self.C, self.H, self.W)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x_hat = self.deconv3(x)
        return x_hat


class ConditionalVAE(nn.Module):
    """
    Pitch-conditioned VAE (conditioning only in the decoder).

    Encoder:
      q(z|x) = N(mu(x), sigma(x))   (same as baseline)

    Decoder:
      p(x|z, pitch) using pitch embedding
    """
    def __init__(
        self,
        latent_dim: int = 32,
        pitch_vocab: int = 128,
        cond_dim: int = 16,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pitch_vocab = pitch_vocab
        self.cond_dim = cond_dim

        self.encoder = VAEConvEncoder(latent_dim=latent_dim)
        self.pitch_cond = PitchConditioner(vocab_size=pitch_vocab, cond_dim=cond_dim)
        self.decoder = ConditionalConvDecoder(latent_dim=latent_dim, cond_dim=cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        pitch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x:     (B, 1, 80, 128)
        pitch: (B,) MIDI in [0..127]

        returns: x_hat, mu, logvar, z
        """
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)

        cond = self.pitch_cond(pitch)          # (B, cond_dim)
        x_hat = self.decoder(z, cond)          # (B, 1, 80, 128)

        return x_hat, mu, logvar, z

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: returns mu(x) only (useful for timbre space plots).
        """
        mu, _ = self.encoder(x)
        return mu
