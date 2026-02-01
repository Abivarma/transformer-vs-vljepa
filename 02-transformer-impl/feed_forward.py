"""
Feed-Forward Network for Transformer.

This module implements the position-wise feed-forward network used in each
layer of the Transformer encoder and decoder.

FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

Reference: "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Consists of two linear transformations with a ReLU activation in between:
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the feed-forward network (typically 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # First linear transformation
        self.linear1 = nn.Linear(d_model, d_ff)

        # Second linear transformation
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for the feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation + ReLU
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Second linear transformation
        x = self.linear2(x)
        x = self.dropout(x)

        return x
