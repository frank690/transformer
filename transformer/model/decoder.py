"""
This module contains the decoder classes of the transformer
"""

__all__ = ["Decoder"]

from typing import Dict, Optional

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
                DecoderLayer(
                    embedding_dimension=embedding_dimension,
                    block_size=block_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        data: torch.Tensor,
        encoded_data: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Define the forward pass behavior of the encoder.
        :param data: input data.
        :param encoded_data: encoded data from the encoder.
        :param padding_mask: padding mask
        :return: dictionary containing decoder and encoder data.
        """
        x = self.word_embedding(data)
        x = self.positional_encoding(x)
        return self.layers(
            {"decoder": x, "encoder": encoded_data, "padding_mask": padding_mask}
        )


class DecoderLayer(nn.Module):
    """
    This class represents a single decoder layer of the transformer.
    """

    def __init__(
        self, embedding_dimension: int, block_size: int, num_heads: int, dropout: float
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
            block_future_tokens=True,
        )
        self.norming_1 = nn.LayerNorm(embedding_dimension)
        self.cross_multi_head = Heads(
            embedding_dimension=embedding_dimension,
            block_size=block_size,
            num_heads=num_heads,
            dropout=dropout,
            block_future_tokens=False,
        )
        self.norming_2 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = FeedForward(
            embedding_dimension=embedding_dimension, dropout=dropout
        )
        self.norming_3 = nn.LayerNorm(embedding_dimension)

    def forward(self, data: Dict) -> Dict:
        """
        Define the forward pass behavior of the encoder layer.
        :param data: dictionary containing input data and encoded data.
        :return: input dictionary but with updated decoder data.
        """
        decoder_data = data["decoder"]
        encoder_data = data["encoder"]
        padding_mask = data["padding_mask"]

        x = self.multi_head(data=decoder_data, padding_mask=padding_mask)
        decoder_data = self.dropout(x)
        decoder_data = self.norming_1(decoder_data)

        y = self.cross_multi_head(data=decoder_data, encoder_data=encoder_data)
        decoder_data = self.dropout(y)
        decoder_data = self.norming_2(decoder_data)

        z = self.feed_forward(decoder_data)
        decoder_data = self.dropout(z)
        decoder_data = self.norming_3(decoder_data)

        return {
            "decoder": decoder_data,
            "encoder": encoder_data,
            "padding_mask": padding_mask,
        }
