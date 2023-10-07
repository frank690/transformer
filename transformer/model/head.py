"""
This module implements the attention heads of the transformer.
"""

__all__ = ["Heads"]

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Heads(nn.Module):
    """
    Class defining multiple heads of the transformer.
    """

    def __init__(
        self,
        embedding_dimension: int,
        block_size: int,
        num_heads: int,
        dropout: float,
        is_masked: bool,
    ) -> None:
        """
        Initialization method.
        :param embedding_dimension: embedding dimension.
        :param block_size: maximum context length for predictions.
        :param num_heads: number of heads.
        :param dropout: dropout rate.
        :param is_masked: whether the heads are masked or not.
        :return: None
        """
        super().__init__()

        head_size = embedding_dimension // num_heads

        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(
            [
                Head(
                    embedding_dimension=embedding_dimension,
                    head_size=head_size,
                    block_size=block_size,
                    dropout=dropout,
                    is_masked=is_masked,
                )
                for _ in range(num_heads)
            ]
        )
        self.linear = nn.Linear(head_size * num_heads, embedding_dimension)

        assert embedding_dimension % num_heads == 0

    def forward(
        self, data: torch.Tensor, encoder_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Define the forward pass behavior of the heads.
        :param data: input data.
        :param encoder_data: encoder data, if any (default: None).
        :return: output data.
        """
        result = torch.cat([head(data, encoder_data) for head in self.heads], dim=-1)
        result = self.linear(result)
        return self.dropout(result)


class Head(nn.Module):
    """
    Class defining a single Head of the transformer.
    """

    def __init__(
        self,
        embedding_dimension: int,
        head_size: int,
        block_size: int,
        dropout: float,
        is_masked: bool,
    ) -> None:
        """
        Initialization method.
        :param embedding_dimension: embedding dimension.
        :param head_size: size of the head (a whole numbered fraction of the embedding dimension).
        :param block_size: maximum context length for predictions.
        :param dropout: dropout rate.
        :param is_masked: whether the head is masked or not.
        :return: None
        """
        super().__init__()
        self.is_masked = is_masked
        self.embedding_dimension = embedding_dimension

        self.dropout = nn.Dropout(dropout)

        self.key = nn.Linear(
            embedding_dimension, head_size, bias=False
        )  # what head contains
        self.query = nn.Linear(
            embedding_dimension, head_size, bias=False
        )  # what head is looking for
        self.value = nn.Linear(
            embedding_dimension, head_size, bias=False
        )  # what head returns to others

        if self.is_masked:
            self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(
        self, data: torch.Tensor, encoder_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Define the forward pass behavior of the head.
        :param data: input data.
        :param encoder_data: encoder data.
        :return: output data.
        """
        if encoder_data is None:
            encoder_data = data

        batch_size, token_size, _ = data.shape
        _, encoder_token_size, embedding_size = encoder_data.shape

        assert (
            embedding_size == self.embedding_dimension
        ), f"Given input embedding dimension ({embedding_size}) should match layer embedding dimension ({self.embedding_dimension})"

        keys = self.key(encoder_data)
        values = self.value(encoder_data)
        queries = self.query(data)

        weights = queries @ keys.transpose(-2, -1) / (embedding_size**0.5)

        assert weights.shape == (
            batch_size,
            token_size,
            encoder_token_size,
        ), f"Weights have shape {weights.shape}, but should have {(batch_size, encoder_token_size, token_size)}."

        if self.is_masked:
            weights = weights.masked_fill(
                self.mask[:encoder_token_size, :token_size] == 0, float("-inf")
            )

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ values
