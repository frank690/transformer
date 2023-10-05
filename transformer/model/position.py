"""
This module contains all the classes and functions related to positional encoding of the data that is fed into the transformer.
"""

__all__ = ["encode"]

from math import log

import torch
import torch.nn as nn

from transformer.config import BATCH_SIZE, BLOCK_SIZE


class PositionalEncoding(nn.Module):
    """Positonal encoding class"""

    def __init__(self, dropout: float) -> None:
        """
        Initialization method.
        :param dropout: dropout rate.
        :return: None
        """
        super().__init__()

        assert BATCH_SIZE % 2 == 0, "Dimension must be an even number"

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(BLOCK_SIZE).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, BATCH_SIZE, 2) * (-log(10000.0) / BATCH_SIZE)
        )
        encoding = torch.zeros(BLOCK_SIZE, 1, BATCH_SIZE)
        encoding[:, 0, 0::2] = torch.sin(position * denominator)
        encoding[:, 0, 1::2] = torch.cos(position * denominator)
        self.register_buffer("encoding", encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass behavior of the positional encoding.
        :param x: input data
        :return: positional encoded output data
        """
        x = x + self.encoding[: x.size(0)]
        return self.dropout(x)
