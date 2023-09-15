"""
This module holds all the configurations related to running the transformer.
"""

__all__ = [
    "DEVICE",
    "TEST_TRAIN_VAL_SPLIT",
    "BLOCK_SIZE",
    "BATCH_SIZE",
]

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_TRAIN_VAL_SPLIT = (0.1, 0.8, 0.1)
BLOCK_SIZE = 32
BATCH_SIZE = 4
