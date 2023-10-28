"""
This moduel contains tests for the FeedForward class.
"""

from contextlib import nullcontext

import pytest
import torch

from transformer.model.feed_forward import FeedForward


class TestFeedForward:
    """
    This class contains all the tests for the FeedForward class.
    """

    @pytest.mark.parametrize(
        "embedding_dimension, dropout, expected_error",
        [
            (42, 0.1, None),
            (1337, -5, ValueError),
            (815, 1.1, ValueError),
            (1, 0.5, None),
        ],
    )
    def test_init(
        self,
        embedding_dimension: int,
        dropout: float,
        expected_error: Exception,
    ):
        """
        Tests the __init__ method.
        """
        with pytest.raises(
            expected_error
        ) if expected_error is not None else nullcontext():
            ff = FeedForward(
                embedding_dimension=embedding_dimension,
                dropout=dropout,
            )

            x = torch.rand(10, 20, embedding_dimension)
            y = ff(x)
            assert y.shape == x.shape
