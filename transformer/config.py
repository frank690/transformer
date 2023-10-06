"""
This module holds all the configurations related to running the transformer.
"""

__all__ = [
    "DEVICE",
    "TEST_TRAIN_VAL_SPLIT",
    "BATCH_SIZE",
    "BLOCK_SIZE",
    "EMBEDDING_DIMENSION",
    "NUMBER_ATTENTION_HEADS",
    "NUMBER_TRANSFORMER_LAYERS",
    "DROPOUT_RATE",
]

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_TRAIN_VAL_SPLIT = (0.1, 0.8, 0.1)
BATCH_SIZE = 64  # how many independent sequences will we process in parallel?
BLOCK_SIZE = 256  # what is the maximum context length for predictions?
EMBEDDING_DIMENSION = 128
NUMBER_ATTENTION_HEADS = 4
NUMBER_TRANSFORMER_LAYERS = 2
DROPOUT_RATE = 0.2
