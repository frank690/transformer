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
        source_vocabulary_size: int,
        target_vocabulary_size: int,
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
            vocabulary_size=source_vocabulary_size,
            embedding_dimension=embedding_dimension,
            block_size=block_size,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=layers,
            vocabulary_size=target_vocabulary_size,
            embedding_dimension=embedding_dimension,
            block_size=block_size,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.linear = nn.Linear(embedding_dimension, target_vocabulary_size)

    def forward(
        self, source_data: torch.Tensor, target_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Define the forward pass behavior of the transformer.
        :param source_data: source data that is fed into the encoder.
        :param target_data: target data that is fed into the decoder.
        :return: output data
        """
        encoded_data = self.encoder(source_data)
        decoded_data = self.decoder(target_data, encoded_data).get("decoder")
        return self.linear(decoded_data)
