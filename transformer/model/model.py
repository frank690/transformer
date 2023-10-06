"""
This module contains the overall transformer class that puts together all subclasses (encoder, decoder, heads, etc.)
"""

__all__ = ["Transformer"]

import torch
import torch.nn as nn

from transformer.config import (
    BLOCK_SIZE,
    DROPOUT_RATE,
    EMBEDDING_DIMENSION,
    NUMBER_ATTENTION_HEADS,
    NUMBER_TRANSFORMER_LAYERS,
)
from transformer.model.decoder import Decoder
from transformer.model.encoder import Encoder


class Transformer(nn.Module):
    """The actual transformer class"""

    def __init__(
        self,
        layers: int = NUMBER_TRANSFORMER_LAYERS,
        embedding_dimension: int = EMBEDDING_DIMENSION,
        block_size: int = BLOCK_SIZE,
        num_attention_heads: int = NUMBER_ATTENTION_HEADS,
        dropout: float = DROPOUT_RATE,
    ) -> None:
        """
        Initialization method.

        :param layers: number of layers.
        :return: None
        """
        super().__init__()

        self.encoder = Encoder(
            num_layers=layers,
            embedding_dimension=embedding_dimension,
            block_size=block_size,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=layers,
            embedding_dimension=embedding_dimension,
            block_size=block_size,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.linear = nn.Linear()  # TODO: what is the output dimension?

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass behavior of the transformer.
        :param data: input data
        :return: output data
        """
        _, encoded_value, encoded_key = self.encoder(data)
        return self.decoder(data, encoded_value, encoded_key)
