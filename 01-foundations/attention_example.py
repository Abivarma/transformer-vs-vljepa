"""Scaled Dot-Product Attention - Working Implementation.

Educational example demonstrating the core attention mechanism.
"""

import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape (batch, seq_len, d_k)
        key: Key tensor of shape (batch, seq_len, d_k)
        value: Value tensor of shape (batch, seq_len, d_v)
        mask: Optional mask tensor (batch, seq_len, seq_len)

    Returns:
        output: Attention output (batch, seq_len, d_v)
        attention_weights: Attention weights (batch, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # Step 1: Compute attention scores (Q @ K^T)
    # Shape: (batch, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k) to prevent gradient vanishing
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask if provided (for padding or causal masking)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Step 4: Apply softmax to get attention weights
    # Shape: (batch, seq_len, seq_len)
    attention_weights = F.softmax(scores, dim=-1)

    # Step 5: Compute weighted sum of values
    # Shape: (batch, seq_len, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


def demonstrate_attention():
    """Demonstrate attention mechanism with a simple example."""
    print("=" * 70)
    print("Scaled Dot-Product Attention Demonstration")
    print("=" * 70)

    # Example parameters
    batch_size = 1
    seq_length = 4
    d_k = 8  # Dimension of queries and keys
    d_v = 8  # Dimension of values

    # Create sample input tensors
    print("\nInput Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Query/Key dimension (d_k): {d_k}")
    print(f"  Value dimension (d_v): {d_v}")

    # Initialize Q, K, V with random values
    torch.manual_seed(42)  # For reproducibility
    query = torch.randn(batch_size, seq_length, d_k)
    key = torch.randn(batch_size, seq_length, d_k)
    value = torch.randn(batch_size, seq_length, d_v)

    print(f"\nQuery shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")

    # Compute attention
    output, attention_weights = scaled_dot_product_attention(query, key, value)

    print("\n" + f"{'Output':-^70}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Display attention weights
    print(f"\n{'Attention Weights (how much each position attends to others)':-^70}")
    print("Each row shows attention distribution for one query position:\n")
    weights = attention_weights[0].detach().numpy()

    for i in range(seq_length):
        print(f"Position {i}: ", end="")
        for j in range(seq_length):
            print(f"{weights[i][j]:.3f} ", end="")
        print(f"  (sum={weights[i].sum():.3f})")

    # Verify attention weights sum to 1
    print(f"\n{'Verification':-^70}")
    row_sums = attention_weights.sum(dim=-1)
    print(
        f"All attention weights sum to 1.0: {torch.allclose(row_sums, torch.ones_like(row_sums))}"
    )

    # Show what happens without scaling
    print(f"\n{'Effect of Scaling':-^70}")
    unscaled_scores = torch.matmul(query, key.transpose(-2, -1))
    scaled_scores = unscaled_scores / math.sqrt(d_k)

    print(f"Unscaled scores variance: {unscaled_scores.var():.4f}")
    print(f"Scaled scores variance: {scaled_scores.var():.4f}")
    print(f"Scaling factor (sqrt(d_k)): {math.sqrt(d_k):.4f}")
    print("\nScaling prevents extreme values that would cause gradient problems!")

    return output, attention_weights


if __name__ == "__main__":
    output, weights = demonstrate_attention()
    print("\n" + "=" * 70)
    print("✓ Attention mechanism executed successfully!")
    print("=" * 70)
