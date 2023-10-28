"""
This moduel contains tests for the PositionalEncoding class.
"""

from contextlib import nullcontext

import pytest
import torch

from transformer.model.position import PositionalEncoding


class TestPositionalEncoding:
    """
    This class contains all the tests for the PositionalEncoding class.
    """

    @pytest.mark.parametrize(
        "embedding_dimension, block_size, dropout, expected_error",
        [
            (42, 128, 0.1, None),
            (42, 128, -5, ValueError),
            (42, 128, 1.1, ValueError),
            (42, 128, 1, None),
            (42, 128, 0.5, None),
            (43, 128, 0.5, AssertionError),
            (42, 63, 0.5, AssertionError),
        ],
    )
    def test_init(
        self,
        embedding_dimension: int,
        block_size: int,
        dropout: float,
        expected_error: Exception,
    ):
        """
        Tests the __init__ method.
        """
        with pytest.raises(
            expected_error
        ) if expected_error is not None else nullcontext():
            pos = PositionalEncoding(
                embedding_dimension=embedding_dimension,
                block_size=block_size,
                dropout=dropout,
            )
            batch_size = 64
            current_max_sentence_length = 222
            vocabulary_size = 1337
            x = torch.randint(
                0,
                vocabulary_size,
                (batch_size, current_max_sentence_length, embedding_dimension),
            )

            assert pos.encoding.shape == (block_size, 1, embedding_dimension)
            assert torch.all(pos.encoding[0, 0, 0::2] == 0)
            assert torch.all(pos.encoding[0, 0, 1::2] == 1)
            assert torch.all(pos.encoding[1, 0, 0::2] != 0)
            assert torch.all(pos.encoding[1, 0, 0::2] != 1)
            assert torch.all((pos.encoding <= 1) & (pos.encoding >= -1))

            assert pos(x).shape == x.shape
