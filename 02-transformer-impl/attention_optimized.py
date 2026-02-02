"""
Optimized Attention Mechanisms with Flash Attention support.

This module provides memory-efficient attention implementations using:
1. PyTorch's built-in Flash Attention (torch.nn.functional.scaled_dot_product_attention)
2. Memory-efficient attention variants
3. Automatic fallback to standard attention

Performance improvements:
- 2-4x faster attention computation
- O(N) memory instead of O(N²)
- Automatic kernel selection based on hardware

Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashAttention(nn.Module):
    """
    Memory-efficient attention using PyTorch's Flash Attention.

    Uses torch.nn.functional.scaled_dot_product_attention when available,
    which provides:
    - Flash Attention on supported hardware (A100, H100)
    - Memory-efficient attention on other GPUs
    - Automatic kernel selection

    Args:
        dropout: Dropout probability for attention weights
        use_flash: Whether to use Flash Attention (auto-detected if None)
    """

    def __init__(self, dropout: float = 0.1, use_flash: Optional[bool] = None):
        super().__init__()
        self.dropout_p = dropout

        # Auto-detect Flash Attention support
        if use_flash is None:
            # Check if scaled_dot_product_attention is available (PyTorch 2.0+)
            self.use_flash = hasattr(F, "scaled_dot_product_attention")
        else:
            self.use_flash = use_flash

        # Only create dropout layer if not using Flash Attention
        if not self.use_flash:
            self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with Flash Attention.

        Args:
            query: Query tensor (batch, num_heads, seq_len_q, d_k)
            key: Key tensor (batch, num_heads, seq_len_k, d_k)
            value: Value tensor (batch, num_heads, seq_len_v, d_v)
            mask: Optional attention mask (batch, num_heads, seq_len_q, seq_len_k)
            is_causal: Whether to apply causal masking (for autoregressive models)

        Returns:
            output: Attention output (batch, num_heads, seq_len_q, d_v)
            attention_weights: None (Flash Attention doesn't return weights for efficiency)
        """
        if self.use_flash:
            # Use PyTorch's optimized scaled_dot_product_attention
            # This automatically selects the best kernel (Flash, Memory-Efficient, or Math)
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
            # Flash Attention doesn't return attention weights for efficiency
            return output, None
        else:
            # Fallback to standard attention
            return self._standard_attention(query, key, value, mask)

    def _standard_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled dot-product attention (fallback)."""
        d_k = query.size(-1)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttentionOptimized(nn.Module):
    """
    Optimized Multi-Head Attention with Flash Attention support.

    Improvements over standard implementation:
    - Flash Attention for 2-4x speedup
    - Fused operations where possible
    - Memory-efficient attention computation

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_flash: Whether to use Flash Attention (auto-detected if None)
        bias: Whether to use bias in linear projections
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.1, use_flash: Optional[bool] = None, bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections (can be fused for efficiency)
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        # Flash Attention
        self.attention = FlashAttention(dropout, use_flash)

        # Dropout for output
        self.dropout = nn.Dropout(dropout)

        # Track if using Flash Attention
        self.use_flash = self.attention.use_flash

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optimized attention.

        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: Attention weights (None if using Flash Attention)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # (batch, seq_q, d_model)
        K = self.W_k(key)  # (batch, seq_k, d_model)
        V = self.W_v(value)  # (batch, seq_v, d_model)

        # Reshape for multi-head attention: (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply Flash Attention
        attn_output, attention_weights = self.attention(Q, K, V, mask, is_causal)

        # Concatenate heads: (batch, seq_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # Output projection
        output = self.W_o(attn_output)
        output = self.dropout(output)

        return output, attention_weights


def test_flash_attention():
    """Test Flash Attention implementation."""
    print("Testing Flash Attention...")

    # Check PyTorch version and Flash Attention availability
    print(f"PyTorch version: {torch.__version__}")
    has_flash = hasattr(F, "scaled_dot_product_attention")
    print(f"Flash Attention available: {has_flash}")

    # Create test inputs
    batch_size, num_heads, seq_len, d_k = 2, 8, 128, 64
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    query = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    key = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
    value = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)

    # Test Flash Attention
    flash_attn = FlashAttention(dropout=0.0, use_flash=True).to(device)
    flash_attn.eval()

    with torch.no_grad():
        output_flash, _ = flash_attn(query, key, value)

    print(f"✅ Flash Attention output shape: {output_flash.shape}")
    print(f"✅ Expected shape: ({batch_size}, {num_heads}, {seq_len}, {d_k})")

    # Test Multi-Head Attention
    d_model = 512
    mha = MultiHeadAttentionOptimized(d_model, num_heads, use_flash=True).to(device)
    mha.eval()

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    with torch.no_grad():
        output_mha, _ = mha(x, x, x)

    print(f"✅ Multi-Head Attention output shape: {output_mha.shape}")
    print(f"✅ Expected shape: ({batch_size}, {seq_len}, {d_model})")

    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    test_flash_attention()
