"""
Temporal Encoder Module (LSTM)
Encodes a borrower's T-month financial trajectory into a fixed-size embedding.

This implements the temporal component of the Spatiotemporal GNN architecture
The LSTM learns sequential
deterioration patterns like:
- Gradual rating decline over 6+ months
- Sudden downgrade spikes before default
- Recovery patterns after temporary stress

Architecture:
    Input:  [N, T, input_dim]  -- N borrowers, T months, input_dim features per month
    Output: [N, hidden_dim]    -- fixed-size temporal embedding per borrower

Reference:
    Chen et al. (2025) - Spatiotemporal Fusion and Selective Aggregation
"""

import torch
import torch.nn as nn
from typing import Optional


class TemporalEncoder(nn.Module):
    """
    LSTM-based temporal encoder for borrower financial trajectories.

    Encodes a sequence of monthly financial features into a fixed-size embedding
    that captures deterioration velocity, trend direction, and volatility.

    Parameters
    ----------
    input_dim : int, default=6
        Number of features per time step.
        Default features: rating_numeric, rating_change_3m, rating_change_6m,
                         rating_change_12m, downgrades_12m, upgrades_12m
    hidden_dim : int, default=16
        LSTM hidden state dimension (also output dimension).
        Kept small (16) for laptop training efficiency.
    num_layers : int, default=1
        Number of stacked LSTM layers.
    dropout : float, default=0.0
        Dropout between LSTM layers (only applies if num_layers > 1).
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Layer norm on output for stable training
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        sequences: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode temporal sequences into fixed-size embeddings.

        Parameters
        ----------
        sequences : torch.Tensor
            Shape [N, T, input_dim] -- padded sequences of monthly features.
            Zero-padded for borrowers with fewer than T months of history.
        lengths : Optional[torch.Tensor]
            Shape [N] -- actual sequence lengths (before padding).
            If provided, uses pack_padded_sequence for efficiency.
            If None, processes all time steps.

        Returns
        -------
        torch.Tensor
            Shape [N, hidden_dim] -- temporal embedding per borrower.
            This is the last hidden state of the LSTM.
        """
        if lengths is not None:
            # Clamp lengths to minimum 1 (borrowers with no history have all-zero
            # sequences, so processing 1 zero step gives zero hidden state)
            clamped_lengths = lengths.clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                sequences, clamped_lengths, batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            # Process all time steps (simpler, works with zero-padded sequences)
            _, (h_n, _) = self.lstm(sequences)

        # h_n shape: [num_layers, N, hidden_dim]
        # Take last layer's hidden state
        output = h_n[-1]  # [N, hidden_dim]

        # Layer norm for training stability
        output = self.layer_norm(output)

        return output

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
