from __future__ import annotations

"""Transformer encoder per stima frame-by-frame del pitch dello stinco.

Input:  sequenza di keypoint 2D normalizzati  (batch, T, 34)
Output: pitch sinistro e destro per ogni frame (batch, T, 2)

Architettura:
  - Linear projection 34 → d_model
  - Positional encoding sinusoidale (learned-free)
  - N × TransformerEncoderLayer (self-attention, feed-forward)
  - Linear head d_model → 2

Il modello è causale (usa solo i frame passati) tramite una maschera
causale nell'attenzione, in modo da essere utilizzabile anche in streaming.
"""

import math

import numpy as np
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding sinusoidale fisso (non appreso)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PitchTransformer(nn.Module):
    """Transformer encoder causale: keypoint 2D → pitch stinco.

    Parameters
    ----------
    input_dim:
        Dimensione input per frame (default 34 = 17 keypoint × 2 coordinate).
    d_model:
        Dimensione interna del Transformer.
    nhead:
        Numero di teste di attenzione (deve dividere d_model).
    num_layers:
        Numero di TransformerEncoderLayer.
    dim_feedforward:
        Dimensione del feed-forward interno.
    dropout:
        Dropout rate.
    output_dim:
        Numero di output per frame (default 2: left_pitch, right_pitch).
    """

    def __init__(
        self,
        input_dim: int = 34,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_dim: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            (B, T, input_dim) — sequenza di keypoint normalizzati.
        src_key_padding_mask:
            (B, T) bool tensor — True dove il frame è padding.
        causal:
            Se True, applica maschera causale (ogni frame vede solo il passato).

        Returns
        -------
        (B, T, output_dim) — pitch predetto per ogni frame.
        """
        B, T, _ = x.shape

        # Proiezione + positional encoding
        x = self.input_proj(x)       # (B, T, d_model)
        x = self.pos_enc(x)

        # Maschera causale
        attn_mask = None
        if causal:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        x = self.encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        return self.head(x)  # (B, T, 2)

    # ── save / load ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "config": {
                "input_dim":       self.input_proj.in_features,
                "d_model":         self.d_model,
                "nhead":           self.encoder.layers[0].self_attn.num_heads,
                "num_layers":      len(self.encoder.layers),
                "dim_feedforward": self.encoder.layers[0].linear1.out_features,
                "dropout":         self.encoder.layers[0].dropout.p,
                "output_dim":      self.head.out_features,
            },
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PitchTransformer":
        ckpt  = torch.load(path, map_location=device)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    @staticmethod
    def best_device() -> str:
        """Restituisce il miglior device disponibile: mps > cuda > cpu."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def predict_numpy(
        self,
        seq: np.ndarray,
        device: str | None = None,
    ) -> np.ndarray:
        """Predice pitch da un array numpy (T, 34) → (T, 2) gradi."""
        if device is None:
            device = self.best_device()
        self.to(device)
        self.eval()
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            out = self.forward(x, causal=True)
        return out.squeeze(0).cpu().numpy()
