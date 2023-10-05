"""
This module contains the decoder classes of the transformer
"""

__all__ = ["Decoder"]

import torch
import torch.nn as nn

from transformer.model.feed_forward import FeedForward
from transformer.model.head import Heads
from transformer.model.position import PositionalEncoding


class Decoder(nn.Module):
    """
    The actual decoder class
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
                DecoderLayer(
                    embedding_dimension=embedding_dimension,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, data: torch.Tensor, encoded_value: torch.Tensor, encoded_key: torch.Tensor
    ) -> torch.Tensor:
        """
        Define the forward pass behavior of the encoder.
        :param data: input data.
        :param encoded_value: encoded value from the encoder.
        :param encoded_key: encoded key from the encoder.
        :return: output data.
        """
        x = self.positional_encoding(data)
        return self.layers(x, encoded_value, encoded_key)


class DecoderLayer(nn.Module):
    """
    This class represents a single decoder layer of the transformer.
    """

    def __init__(
        self, embedding_dimension: int, num_heads: int, dropout: float
    ) -> None:
        """
        Initialization method.
        :param embedding_dimension: embedding dimension.
        :param num_heads: number of attention heads.
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.masked_multi_head = Heads(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            dropout=dropout,
            is_masked=True,
        )
        self.norming_1 = nn.LayerNorm(embedding_dimension)
        self.cross_multi_head = Heads(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norming_2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = FeedForward(
            embedding_dimension=embedding_dimension, dropout=dropout
        )
        self.norming_3 = nn.LayerNorm(embedding_dimension)

    def forward(self, data: torch.Tensor, encoded_data: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass behavior of the encoder layer.
        :param data: input data.
        :param encoded_data: encoded data from the encoder.
        :return: output data.
        """
        x = self.masked_multi_head(data)
        data += self.dropout(x)
        data = self.norming_1(data)

        y = self.cross_multi_head(
            data, encoded_data
        )  # TODO: add input from encoder here somehow
        data += self.dropout(y)
        data = self.norming_2(data)

        z = self.feed_forward(data)
        data += self.dropout(z)
        return self.norming_3(data)
