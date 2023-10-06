"""
This module contains the feed forward sublayer of the transformer.
"""

__all__ = ["FeedForward"]

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Class defining the feed forward sublayer of the transformer.
    """

    def __init__(self, embedding_dimension: int, dropout: float) -> None:
        """
        Initialize the feed forward class.
        :param embedding_dimension: embedding dimension.
        :param dropout: dropout rate.
        :return: None
        """

        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(embedding_dimension, 2 * embedding_dimension),
            nn.ReLU(),
            nn.Linear(2 * embedding_dimension, embedding_dimension),
            nn.Dropout(dropout),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass behavior of the feed forward sublayer.
        :param data: input data
        :return: output data
        """
        return self.linear(data)
