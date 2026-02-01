"""
Transformer Encoder.

This module implements the encoder component of the Transformer architecture.
The encoder consists of a stack of N identical layers, each with two sub-layers:
1. Multi-head self-attention
2. Position-wise feed-forward network

Each sub-layer has a residual connection and layer normalization.

Reference: "Attention is All You Need" (Vaswani et al., 2017)
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
    from .feed_forward import FeedForward
except ImportError:
    from attention import MultiHeadAttention
    from feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer.

    Consists of:
    1. Multi-head self-attention mechanism
    2. Position-wise feed-forward network
    Each with residual connections and layer normalization.

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Forward pass for encoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention sub-layer with residual connection and layer norm
        # Pre-norm architecture: norm -> sublayer -> residual
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward sub-layer with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder Stack.

    Stacks N encoder layers together.

    Args:
        num_layers: Number of encoder layers
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout probability
    """

    def __init__(
        self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()

        # Create stack of N encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Forward pass through all encoder layers.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)

        # Final layer normalization
        x = self.norm(x)

        return x
