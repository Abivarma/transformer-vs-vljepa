"""
Complete Transformer Model.

This module implements the full Transformer architecture combining:
- Token embeddings
- Positional encodings
- Encoder stack
- Decoder stack
- Output projection

Reference: "Attention is All You Need" (Vaswani et al., 2017)
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    from .decoder import TransformerDecoder
    from .encoder import TransformerEncoder
    from .positional_encoding import PositionalEncoding
except ImportError:
    from decoder import TransformerDecoder
    from encoder import TransformerEncoder
    from positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    """
    Complete Transformer Model.

    Args:
        src_vocab_size: Size of source vocabulary
        tgt_vocab_size: Size of target vocabulary
        d_model: Dimension of the model (default: 512)
        num_heads: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 6)
        num_decoder_layers: Number of decoder layers (default: 6)
        d_ff: Dimension of feed-forward network (default: 2048)
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Source and target embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encodings
        self.src_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder and decoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)

        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask: Optional[torch.Tensor] = None):
        """
        Encode source sequence.

        Args:
            src: Source sequence of shape (batch_size, src_seq_len)
            src_mask: Optional mask for source sequence

        Returns:
            Encoder output of shape (batch_size, src_seq_len, d_model)
        """
        # Embed and add positional encoding
        src = self.src_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.src_pos_encoding(src)

        # Pass through encoder
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt,
        encoder_output,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        """
        Decode target sequence.

        Args:
            tgt: Target sequence of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional mask for source sequence
            tgt_mask: Optional mask for target sequence (causal mask)

        Returns:
            Decoder output of shape (batch_size, tgt_seq_len, d_model)
        """
        # Embed and add positional encoding
        tgt = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt = self.tgt_pos_encoding(tgt)

        # Pass through decoder
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def forward(
        self,
        src,
        tgt,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through the complete Transformer.

        Args:
            src: Source sequence of shape (batch_size, src_seq_len)
            tgt: Target sequence of shape (batch_size, tgt_seq_len)
            src_mask: Optional mask for source sequence
            tgt_mask: Optional mask for target sequence (causal mask)

        Returns:
            Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Encode source sequence
        encoder_output = self.encode(src, src_mask)

        # Decode target sequence
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary size
        output = self.output_projection(decoder_output)

        return output

    @staticmethod
    def create_causal_mask(size: int, device=None) -> torch.Tensor:
        """
        Create a causal (look-ahead) mask to prevent attention to future positions.

        Args:
            size: Size of the square mask
            device: Device to create the mask on

        Returns:
            Causal mask of shape (1, 1, size, size)
            1 = allow attention, 0 = block attention
        """
        # Create lower triangular matrix (1s below and on diagonal, 0s above)
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def create_padding_mask(seq, pad_idx: int = 0) -> torch.Tensor:
        """
        Create a padding mask to prevent attention to padding tokens.

        Args:
            seq: Input sequence of shape (batch_size, seq_len)
            pad_idx: Index of padding token

        Returns:
            Padding mask of shape (batch_size, 1, 1, seq_len)
        """
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask


class TransformerClassifier(nn.Module):
    """
    Transformer model for sequence classification (e.g., sentiment analysis).

    This is a simplified version that uses only the encoder for classification tasks.

    Args:
        vocab_size: Size of vocabulary
        num_classes: Number of output classes
        d_model: Dimension of the model (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of encoder layers (default: 4)
        d_ff: Dimension of feed-forward network (default: 1024)
        max_len: Maximum sequence length (default: 512)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Forward pass for classification.

        Args:
            x: Input sequence of shape (batch_size, seq_len)
            mask: Optional mask for input sequence

        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Embed and add positional encoding
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)

        # Pass through encoder
        x = self.encoder(x, mask)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, d_model)

        # Classification
        logits = self.classifier(x)  # (batch_size, num_classes)

        return logits

    @staticmethod
    def create_padding_mask(seq, pad_idx: int = 0) -> torch.Tensor:
        """
        Create a padding mask to prevent attention to padding tokens.

        Args:
            seq: Input sequence of shape (batch_size, seq_len)
            pad_idx: Index of padding token

        Returns:
            Padding mask of shape (batch_size, 1, 1, seq_len)
        """
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
