"""
This module contains the encoder classes of the transformer
"""

__all__ = ["Encoder"]

import torch
import torch.nn as nn

from transformer.model.feed_forward import FeedForward
from transformer.model.head import Heads
from transformer.model.position import PositionalEncoding


class Encoder(nn.Module):
    """
    The actual encoder class
    """

    def __init__(
        self, num_layers: int, embedding_dimension: int, num_heads: int, dropout: float
    ) -> None:
        """
        Initialization method.

        :param num_layers: number of layers.
        :param embedding_dimension: embedding dimension.
        :param num_heads: number of attention heads.
        :param dropout: dropout rate.
        :return: None
        """
        super().__init__()

        self.positional_encoding = PositionalEncoding(dropout=dropout)
        self.layers = nn.Sequential(
            *[
                EncoderLayer(
                    embedding_dimension=embedding_dimension,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass behavior of the encoder.
        :param data: input data
        :return: output data
        """
        x = self.positional_encoding(data)
        return self.layers(x)


class EncoderLayer(nn.Module):
    """
    This class represents a single encoder layer of the transformer.
    """

    def __init__(
        self, embedding_dimension: int, num_heads: int, dropout: float
    ) -> None:
        """
        Initialization method.
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.multi_head = Heads(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norming_1 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = FeedForward(
            embedding_dimension=embedding_dimension, dropout=dropout
        )
        self.norming_2 = nn.LayerNorm(embedding_dimension)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass behavior of the encoder layer.
        :param data: input data
        :return: output data
        """
        x = self.multi_head(data)
        x = data + self.dropout(x)
        x = self.norming_1(x)
        x = self.feed_forward(x)
        x = data + self.dropout(x)
        return self.norming_2(x)
