"""
Transformer Decoder.

This module implements the decoder component of the Transformer architecture.
The decoder consists of a stack of N identical layers, each with three sub-layers:
1. Masked multi-head self-attention
2. Multi-head cross-attention (attending to encoder output)
3. Position-wise feed-forward network

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


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer.

    Consists of:
    1. Masked multi-head self-attention
    2. Multi-head cross-attention
    3. Position-wise feed-forward network
    Each with residual connections and layer normalization.

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Cross-attention (encoder-decoder attention)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        encoder_output,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for decoder layer.

        Args:
            x: Input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional mask for source sequence
            tgt_mask: Optional mask for target sequence (causal mask)

        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention sub-layer
        self_attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Cross-attention sub-layer (query from decoder, key & value from encoder)
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder Stack.

    Stacks N decoder layers together.

    Args:
        num_layers: Number of decoder layers
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout probability
    """

    def __init__(
        self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()

        # Create stack of N decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x,
        encoder_output,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through all decoder layers.

        Args:
            x: Input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional mask for source sequence
            tgt_mask: Optional mask for target sequence (causal mask)

        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, d_model)
        """
        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # Final layer normalization
        x = self.norm(x)

        return x
