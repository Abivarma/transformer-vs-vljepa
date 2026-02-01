"""
Comprehensive tests for Transformer implementation.

Tests all components:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Feed-Forward Network
- Encoder Layer & Stack
- Decoder Layer & Stack
- Complete Transformer
- Transformer Classifier
"""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "02-transformer-impl"))

from attention import MultiHeadAttention, ScaledDotProductAttention
from decoder import DecoderLayer, TransformerDecoder
from encoder import EncoderLayer, TransformerEncoder
from feed_forward import FeedForward
from positional_encoding import PositionalEncoding
from transformer import Transformer, TransformerClassifier


class TestScaledDotProductAttention:
    """Test Scaled Dot-Product Attention mechanism."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64
        attention = ScaledDotProductAttention()

        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)

        output, attn_weights = attention(query, key, value)

        assert output.shape == (batch_size, num_heads, seq_len, d_k)
        assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 across the key dimension."""
        batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64
        attention = ScaledDotProductAttention(dropout=0.0)  # No dropout for test

        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)

        _, attn_weights = attention(query, key, value)

        # Sum across key dimension should be 1
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_masking(self):
        """Test that masking prevents attention to certain positions."""
        batch_size, num_heads, seq_len, d_k = 2, 8, 10, 64
        attention = ScaledDotProductAttention(dropout=0.0)

        query = torch.randn(batch_size, num_heads, seq_len, d_k)
        key = torch.randn(batch_size, num_heads, seq_len, d_k)
        value = torch.randn(batch_size, num_heads, seq_len, d_k)

        # Create mask that blocks attention to last 3 positions
        mask = torch.ones(batch_size, 1, seq_len, seq_len)
        mask[:, :, :, -3:] = 0

        _, attn_weights = attention(query, key, value, mask)

        # Attention weights for masked positions should be near zero
        assert torch.all(attn_weights[:, :, :, -3:] < 1e-5)


class TestMultiHeadAttention:
    """Test Multi-Head Attention mechanism."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_model = 2, 10, 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)

        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        output, _ = mha(query, key, value)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths for query and key/value."""
        batch_size, seq_len_q, seq_len_kv, d_model = 2, 5, 10, 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)

        query = torch.randn(batch_size, seq_len_q, d_model)
        key = torch.randn(batch_size, seq_len_kv, d_model)
        value = torch.randn(batch_size, seq_len_kv, d_model)

        output, attn_weights = mha(query, key, value)

        assert output.shape == (batch_size, seq_len_q, d_model)
        assert attn_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_kv)


class TestPositionalEncoding:
    """Test Positional Encoding."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_model = 2, 10, 512
        pe = PositionalEncoding(d_model)

        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_deterministic(self):
        """Test that positional encoding is deterministic."""
        d_model, max_len = 512, 100
        pe = PositionalEncoding(d_model, max_len, dropout=0.0)

        x1 = torch.randn(1, 50, d_model)
        x2 = x1.clone()

        output1 = pe(x1)
        output2 = pe(x2)

        assert torch.allclose(output1, output2)


class TestFeedForward:
    """Test Feed-Forward Network."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_model = 2, 10, 512
        d_ff = 2048
        ff = FeedForward(d_model, d_ff)

        x = torch.randn(batch_size, seq_len, d_model)
        output = ff(x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestEncoderLayer:
    """Test Encoder Layer."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_model = 2, 10, 512
        num_heads, d_ff = 8, 2048
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

        x = torch.randn(batch_size, seq_len, d_model)
        output = encoder_layer(x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestTransformerEncoder:
    """Test Transformer Encoder Stack."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len, d_model = 2, 10, 512
        num_layers, num_heads, d_ff = 6, 8, 2048
        encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff)

        x = torch.randn(batch_size, seq_len, d_model)
        output = encoder(x)

        assert output.shape == (batch_size, seq_len, d_model)


class TestDecoderLayer:
    """Test Decoder Layer."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, tgt_len, src_len, d_model = 2, 8, 10, 512
        num_heads, d_ff = 8, 2048
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff)

        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        output = decoder_layer(x, encoder_output)

        assert output.shape == (batch_size, tgt_len, d_model)


class TestTransformerDecoder:
    """Test Transformer Decoder Stack."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, tgt_len, src_len, d_model = 2, 8, 10, 512
        num_layers, num_heads, d_ff = 6, 8, 2048
        decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff)

        x = torch.randn(batch_size, tgt_len, d_model)
        encoder_output = torch.randn(batch_size, src_len, d_model)
        output = decoder(x, encoder_output)

        assert output.shape == (batch_size, tgt_len, d_model)


class TestTransformer:
    """Test complete Transformer model."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, src_len, tgt_len = 2, 10, 8
        src_vocab_size, tgt_vocab_size = 5000, 5000
        d_model, num_heads = 512, 8
        num_encoder_layers, num_decoder_layers = 6, 6
        d_ff = 2048

        model = Transformer(
            src_vocab_size,
            tgt_vocab_size,
            d_model,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            d_ff,
        )

        src = torch.randint(0, src_vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

        output = model(src, tgt)

        assert output.shape == (batch_size, tgt_len, tgt_vocab_size)

    def test_causal_mask(self):
        """Test causal mask creation."""
        size = 5
        mask = Transformer.create_causal_mask(size)

        assert mask.shape == (1, 1, size, size)
        # Causal mask should be lower triangular (1s below diagonal, 0s above)
        # Upper triangle (diagonal=1) should be 0 (no future attention)
        assert torch.all(torch.triu(mask.squeeze(), diagonal=1) == 0)

    def test_padding_mask(self):
        """Test padding mask creation."""
        batch_size, seq_len = 2, 10
        pad_idx = 0

        seq = torch.tensor(
            [
                [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],  # 4 tokens + 6 padding
                [1, 2, 3, 4, 5, 6, 7, 0, 0, 0],  # 7 tokens + 3 padding
            ]
        )

        mask = Transformer.create_padding_mask(seq, pad_idx)

        assert mask.shape == (batch_size, 1, 1, seq_len)
        # Check that padding positions are masked
        assert mask[0, 0, 0, 4:].sum() == 0
        assert mask[1, 0, 0, 7:].sum() == 0


class TestTransformerClassifier:
    """Test Transformer Classifier for sequence classification."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        batch_size, seq_len = 2, 10
        vocab_size, num_classes = 5000, 2
        d_model, num_heads, num_layers = 256, 8, 4
        d_ff = 1024

        model = TransformerClassifier(vocab_size, num_classes, d_model, num_heads, num_layers, d_ff)

        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = model(x)

        assert output.shape == (batch_size, num_classes)

    def test_padding_mask(self):
        """Test padding mask creation for classifier."""
        batch_size, seq_len = 2, 10
        pad_idx = 0

        seq = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 0, 0, 0]])

        mask = TransformerClassifier.create_padding_mask(seq, pad_idx)

        assert mask.shape == (batch_size, 1, 1, seq_len)


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_transformer_forward_backward(self):
        """Test that gradients flow correctly through the model."""
        batch_size, src_len, tgt_len = 2, 10, 8
        src_vocab_size, tgt_vocab_size = 100, 100
        d_model = 64  # Small for speed

        model = Transformer(
            src_vocab_size,
            tgt_vocab_size,
            d_model=d_model,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=256,
        )

        src = torch.randint(0, src_vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

        # Forward pass
        output = model(src, tgt)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_classifier_training_step(self):
        """Test a single training step of the classifier."""
        batch_size, seq_len = 2, 10
        vocab_size, num_classes = 100, 2
        d_model = 64  # Small for speed

        model = TransformerClassifier(
            vocab_size, num_classes, d_model=d_model, num_heads=4, num_layers=2, d_ff=256
        )

        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, num_classes, (batch_size,))

        # Forward pass
        logits = model(x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        # Backward pass
        loss.backward()

        # Check that loss is a scalar and gradients exist
        assert loss.dim() == 0
        for param in model.parameters():
            assert param.grad is not None


def test_all_imports():
    """Test that all modules can be imported correctly."""
    from attention import MultiHeadAttention, ScaledDotProductAttention
    from decoder import DecoderLayer, TransformerDecoder
    from encoder import EncoderLayer, TransformerEncoder
    from feed_forward import FeedForward
    from positional_encoding import PositionalEncoding
    from transformer import Transformer, TransformerClassifier

    assert ScaledDotProductAttention is not None
    assert MultiHeadAttention is not None
    assert PositionalEncoding is not None
    assert FeedForward is not None
    assert EncoderLayer is not None
    assert TransformerEncoder is not None
    assert DecoderLayer is not None
    assert TransformerDecoder is not None
    assert Transformer is not None
    assert TransformerClassifier is not None
