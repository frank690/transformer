"""
This module contains all the classes and functions related to positional encoding of the data that is fed into the transformer.
"""

__all__ = ["encode"]

from math import log

import torch
import torch.nn as nn

from transformer.config import MAX_NUMBER_OF_TOKENS


class PositionalEncoding(nn.Module):
    """Positonal encoding class"""

    def __init__(self, dimension: int):
        """
        Initialization method.
        :param dimension: dimension of the positional encoding
        """
        super().__init__()

        position = torch.arange(MAX_NUMBER_OF_TOKENS).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, dimension, 2) * (-log(10000.0) / dimension)
        )
        pe = torch.zeros(MAX_NUMBER_OF_TOKENS, 1, dimension)
        pe[:, 0, 0::2] = torch.sin(position * denominator)
        pe[:, 0, 1::2] = torch.cos(position * denominator)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass behavior of the positional encoding.
        :param x: input data
        :return: positional encoded output data
        """
        return x + self.pe[: x.size(0)]
