"""
This module contains the coder class of the transformer which can either be an encoder or a decoder.
"""

__all__ = ["Coder"]

import torch
import torch.nn as nn
import torch.nn.functional as F


class Coder(nn.Module):
    """
    The actual Coder class
    """

    def __init__(self) -> None:
        """
        Initialization method.
        """
        super().__init__()
