"""
This module contains miscellaneous functions used across this project.
"""

__all__ = ["generate_mask"]

import torch


def generate_mask(data: torch.Tensor, padding_index: int = 1) -> torch.Tensor:
    """
    Generate a mask for the given data tensor.
    The given padding_index is used to determine the padding elements.
    :param data: data tensor
    :param padding_index: padding index
    :return: mask tensor
    """
    return (data != padding_index).int().detach()
