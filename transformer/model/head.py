"""
This module implements the attention heads of the transformer.
"""

__all__ = ["Heads"]

import torch
import torch.nn as nn
import torch.nn.functional as F


class Heads(nn.Module):
    """
    Class defining multiple heads of the transformer.
    """

    def __init__(self, size: int, num_heads: int, is_in_encoder: bool = True) -> None:
        """
        Initialization method.
        :param size: size of each head.
        :param num_heads: number of heads.
        :param is_in_encoder: whether the heads are part of an encoder or decoder.
        :return: None
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(size, is_in_encoder) for _ in range(num_heads)]
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass behavior of the heads.
        :param data: input data
        :return: output data
        """
        return torch.cat([head(data) for head in self.heads], dim=-1)


class Head(nn.Module):
    """
    Class defining a single Head of the transformer.
    """

    def __init__(self, size: int, is_in_encoder: bool = True) -> None:
        """
        Initialization method.
        :param size: size of the head
        :param is_in_encoder: whether the head is part of an encoder or decoder
        :return: None
        """
        super().__init__()
        self.is_in_encoder = is_in_encoder

        self.key = nn.Linear(size, size, bias=False)  # what head contains
        self.query = nn.Linear(size, size, bias=False)  # what head is looking for
        self.value = nn.Linear(size, size, bias=False)  # what head returns to others

        self.register_buffer("mask", torch.tril(torch.ones(size, size)))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass behavior of the head.
        :param data: input data
        :return: output data
        """
        BATCH_SIZE, TOKEN_SIZE, CHANNEL_SIZE = data.shape

        keys = self.key(data)
        queries = self.query(data)
        values = self.value(data)

        weights = queries @ keys.transpose(-2, -1) / (CHANNEL_SIZE**0.5)

        if self.is_in_encoder:
            weights = weights.masked_fill(
                self.mask[:TOKEN_SIZE, :TOKEN_SIZE] == 0, float("-inf")
            )

        weights = F.softmax(weights, dim=-1)
        return weights @ values
