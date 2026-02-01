"""
Attention Mechanisms for Transformer.

This module implements:
1. Scaled Dot-Product Attention
2. Multi-Head Attention

Reference: "Attention is All You Need" (Vaswani et al., 2017)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        dropout: Dropout probability for attention weights
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len_q, d_k)
            key: Key tensor of shape (batch_size, num_heads, seq_len_k, d_k)
            value: Value tensor of shape (batch_size, num_heads, seq_len_v, d_v)
            mask: Optional mask tensor to prevent attention to certain positions
                  Shape: (batch_size, 1, seq_len_q, seq_len_k) or broadcastable

        Returns:
            output: Attention output of shape (batch_size, num_heads, seq_len_q, d_v)
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # Get dimension for scaling
        d_k = query.size(-1)

        # Compute attention scores: QK^T / sqrt(d_k)
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (if provided) - set masked positions to large negative value
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        # Shape: (batch_size, num_heads, seq_len_q, d_v)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Instead of performing a single attention function, we project the queries,
    keys, and values h times with different learned linear projections.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the heads back into a single tensor.

        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, d_k)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        # Transpose to (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)
        # Reshape to (batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor

        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 1. Linear projections
        Q = self.W_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)  # (batch_size, seq_len_k, d_model)
        V = self.W_v(value)  # (batch_size, seq_len_v, d_model)

        # 2. Split into multiple heads
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len_v, d_k)

        # 3. Apply attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        # attn_output: (batch_size, num_heads, seq_len_q, d_k)

        # 4. Combine heads
        output = self.combine_heads(attn_output)
        # output: (batch_size, seq_len_q, d_model)

        # 5. Final linear projection
        output = self.W_o(output)
        output = self.dropout(output)

        return output, attention_weights
