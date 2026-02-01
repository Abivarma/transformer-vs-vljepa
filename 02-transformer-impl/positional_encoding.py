"""
Positional Encoding for Transformer.

Since the Transformer contains no recurrence and no convolution, positional
encodings are added to give the model information about the relative or absolute
position of tokens in the sequence.

Reference: "Attention is All You Need" (Vaswani et al., 2017)
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    where pos is the position and i is the dimension.

    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (max_len, d_model) for positional encodings
        pe = torch.zeros(max_len, d_model)

        # Create position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Shape: (max_len, 1)

        # Create division term for the exponential
        # div_term = 1 / (10000^(2i/d_model)) = exp(-log(10000) * 2i / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Shape: (d_model/2,)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but should be saved with the model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added, same shape as input
        """
        # Add positional encoding to input
        # self.pe[:, :x.size(1), :] selects the encodings for the sequence length
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
