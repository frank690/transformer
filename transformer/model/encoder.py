"""
This module contains the encoder classes of the transformer
"""

__all__ = ["Encoder"]

from typing import Optional

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
        self,
        num_layers: int,
        vocabulary_size: int,
        embedding_dimension: int,
        block_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        """
        Initialization method.

        :param num_layers: number of layers.
        :param vocabulary_size: size of the vocabulary.
        :param embedding_dimension: embedding dimension.
        :param block_size: maximum context length for predictions.
        :param num_heads: number of attention heads.
        :param dropout: dropout rate.
        :return: None
        """
        super().__init__()

        self.word_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dimension,
        )
        self.positional_encoding = PositionalEncoding(dropout=dropout)
        self.layers = nn.Sequential(
            *[
                EncoderLayer(
                    embedding_dimension=embedding_dimension,
                    block_size=block_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, data: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Define the forward pass behavior of the encoder.
        :param data: input data
        :param padding_mask: padding mask
        :return: output data
        """
        x = self.word_embedding(data)
        x = self.positional_encoding(x)
        return self.layers(data=x, padding_mask=padding_mask)


class EncoderLayer(nn.Module):
    """
    This class represents a single encoder layer of the transformer.
    """

    def __init__(
        self,
        embedding_dimension: int,
        block_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        """
        Initialization method.

        :param embedding_dimension: embedding dimension.
        :param block_size: maximum context length for predictions.
        :param num_heads: number of attention heads.
        :param dropout: dropout rate.
        :return: None
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.multi_head = Heads(
            embedding_dimension=embedding_dimension,
            block_size=block_size,
            num_heads=num_heads,
            dropout=dropout,
            block_future_tokens=False,
        )
        self.norming_1 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = FeedForward(
            embedding_dimension=embedding_dimension, dropout=dropout
        )
        self.norming_2 = nn.LayerNorm(embedding_dimension)

    def forward(
        self, data: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Define the forward pass behavior of the encoder layer.
        :param data: input data
        :param padding_mask: padding mask
        :return: output data
        """
        x = self.multi_head(data=data, padding_mask=padding_mask)
        x = data + self.dropout(x)
        x = self.norming_1(x)
        x = self.feed_forward(x)
        x = data + self.dropout(x)
        return self.norming_2(x)
