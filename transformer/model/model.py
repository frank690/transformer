"""
This module contains the encoder and decoder classes of the transformer
"""

__all__ = ["Encoder", "Decoder"]

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.config import MAX_NUMBER_OF_TOKENS
from transformer.model.head import Heads
from transformer.model.position import PositionalEncoding


class Encoder(nn.Module):
    """
    The actual encoder class
    """

    def __init__(self, layers: int = 1) -> None:
        """
        Initialization method.

        :param layers: number of layers.
        Each layer consists of a multi attention head and a feed forward layer.
        skip and norming is performed twice as well.
        :return: None
        """
        super().__init__()

        self.positional_encoding = PositionalEncoding()
        self.layers = nn.Sequential(*[EncoderLayer() for _ in range(layers)])

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

    def __init__(self) -> None:
        """
        Initialization method.
        """
        super().__init__()

        self.multi_head = Heads(size=MAX_NUMBER_OF_TOKENS, num_heads=8)

        self.norming = nn.LayerNorm(MAX_NUMBER_OF_TOKENS)

        self.feed_forward = nn.Sequential(
            nn.Linear(MAX_NUMBER_OF_TOKENS, 2 * MAX_NUMBER_OF_TOKENS),
            nn.ReLU(),
            nn.Linear(2 * MAX_NUMBER_OF_TOKENS, MAX_NUMBER_OF_TOKENS),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass behavior of the encoder layer.
        :param data: input data
        :return: output data
        """
        x = self.multi_head(data)
        x = data + x
        x = self.norming(x)
        x = self.feed_forward(x)
        x = x + data
        return self.norming(x)
