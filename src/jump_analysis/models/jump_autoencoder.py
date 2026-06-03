from __future__ import annotations

"""LSTM Autoencoder per anomaly detection su sequenze di drop jump.

Input:  (B, T, 36) — 34 keypoint normalizzati + 2 pitch predetti dal PitchTransformer
Output: (B, T, 36) — ricostruzione della sequenza

Il punteggio di anomalia è l'errore MSE di ricostruzione medio per frame.
La soglia viene calibrata sul 99° percentile degli errori sui dati normali (183 atleti mocap).

Architettura:
  Encoder: LSTM bidirezionale → mean pooling → Linear → latent (z)
  Decoder: Linear → z ripetuto T volte → LSTM unidirezionale → Linear → ricostruzione

Save/load include: pesi del modello, soglia anomalia, statistiche di normalizzazione.
"""

import numpy as np
import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, latent_dim)  # ×2 per bidirezionale

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, input_dim)
        out, _ = self.lstm(x)  # (B, T, hidden*2)
        if mask is not None:
            # Mean pooling solo sui frame validi
            mask_f = (~mask).unsqueeze(-1).float()   # (B, T, 1)
            pooled = (out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            pooled = out.mean(1)   # (B, hidden*2)
        return self.proj(pooled)   # (B, latent_dim)


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.expand = nn.Linear(latent_dim, hidden_dim)
        self.lstm   = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, T: int) -> torch.Tensor:
        # z: (B, latent_dim)
        h = self.expand(z)               # (B, hidden_dim)
        h = h.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden_dim)
        out, _ = self.lstm(h)            # (B, T, hidden_dim)
        return self.proj(out)            # (B, T, output_dim)


class JumpAutoencoder(nn.Module):
    """LSTM Autoencoder per sequenze di drop jump.

    Parameters
    ----------
    input_dim:   Dimensione per frame (default 36 = 34 keypoint + 2 pitch).
    hidden_dim:  Dimensione LSTM interna.
    latent_dim:  Dimensione del vettore latente (bottleneck).
    num_layers:  Strati LSTM in encoder e decoder.
    dropout:     Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 36,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, num_layers, dropout)

        # Statistiche di normalizzazione (impostate da fit_normalization)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std",  torch.ones(input_dim))

        # Soglia di anomalia (percentile 95 sui normali)
        self.anomaly_threshold: float | None = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:    (B, T, input_dim)
        mask: (B, T) bool — True = frame di padding da ignorare
        Returns: ricostruzione (B, T, input_dim)
        """
        z    = self.encoder(x, mask)     # (B, latent_dim)
        recon = self.decoder(z, x.size(1))
        return recon

    # ── normalizzazione ────────────────────────────────────────────────────────

    def fit_normalization(self, sequences: np.ndarray, lengths: np.ndarray) -> None:
        """Calcola media e std solo sui frame validi di tutte le sequenze."""
        frames = []
        for i, L in enumerate(lengths):
            frames.append(sequences[i, :L])
        all_frames = np.concatenate(frames, axis=0)
        mean = all_frames.mean(axis=0).astype(np.float32)
        std  = all_frames.std(axis=0).astype(np.float32)
        std  = np.where(std < 1e-6, 1.0, std)
        self.input_mean = torch.tensor(mean)
        self.input_std  = torch.tensor(std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean.to(x.device)) / self.input_std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.input_std.to(x.device) + self.input_mean.to(x.device)

    # ── anomaly score ──────────────────────────────────────────────────────────

    def reconstruction_error(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        percentile: float = 95.0,
    ) -> torch.Tensor:
        """Anomaly score per sequenza (B,): percentile degli errori per frame.

        Usa il percentile invece della media in modo che anomalie brevi ma intense
        (es. pochi frame con movimento molto anomalo) vengano rilevate.
        Un frame anomalo che dura il 5% della sequenza viene catturato al 95°
        percentile, mentre verrebbe quasi annullato dalla media.
        """
        x_norm = self.normalize(x)
        recon  = self.forward(x_norm, mask)
        err    = (x_norm - recon) ** 2          # (B, T, D)
        per_frame = err.mean(dim=-1)             # (B, T) — MSE medio per frame

        # Percentile per ogni sequenza del batch
        scores = torch.zeros(x.size(0), device=x.device)
        for b in range(x.size(0)):
            valid_errors = per_frame[b]
            if mask is not None:
                valid_errors = valid_errors[~mask[b]]   # esclude padding
            if valid_errors.numel() == 0:
                scores[b] = 0.0
            else:
                k = max(1, int(valid_errors.numel() * percentile / 100.0))
                scores[b] = valid_errors.topk(k).values.mean()
        return scores

    def anomaly_score_numpy(
        self,
        seq: np.ndarray,
        device: str | None = None,
    ) -> float:
        """Calcola anomaly score per una singola sequenza numpy (T, input_dim)."""
        if device is None:
            device = self.best_device()
        self.to(device)
        self.eval()
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            score = self.reconstruction_error(x)
        return float(score.item())

    def frame_errors_numpy(
        self,
        seq: np.ndarray,
        device: str | None = None,
    ) -> np.ndarray:
        """Errore MSE per frame (T,) — usato per capire QUANDO il movimento è anomalo."""
        if device is None:
            device = self.best_device()
        self.to(device)
        self.eval()
        with torch.no_grad():
            x     = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            x_norm = self.normalize(x)
            recon  = self.forward(x_norm)
            err    = (x_norm - recon) ** 2    # (1, T, D)
            per_frame = err.mean(dim=-1)       # (1, T)
        return per_frame.squeeze(0).cpu().numpy()   # (T,)

    def is_anomaly(self, seq: np.ndarray, device: str | None = None) -> tuple[bool, float]:
        """Restituisce (is_anomaly, score). Richiede anomaly_threshold impostato."""
        if self.anomaly_threshold is None:
            raise RuntimeError("anomaly_threshold non impostato. Esegui prima train_jump_autoencoder.py.")
        score = self.anomaly_score_numpy(seq, device=device)
        return score > self.anomaly_threshold, score

    # ── save / load ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "config": {
                "input_dim":  self.input_dim,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "num_layers": self.num_layers,
                "dropout":    0.0,  # non serve a inference
            },
            "state_dict":        self.state_dict(),
            "anomaly_threshold": self.anomaly_threshold,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "JumpAutoencoder":
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.anomaly_threshold = ckpt.get("anomaly_threshold")
        model.eval()
        return model

    @staticmethod
    def best_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
