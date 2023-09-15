"""
This module contains all the constants used throughout this project.
"""

__all__ = [
    "DATA_DOWNLOAD_SOURCE",
    "ZIPPED_DATA_PATH",
    "DE_DATA_PATH",
    "EN_DATA_PATH",
]


DATA_DOWNLOAD_SOURCE = "https://www.statmt.org/europarl/v7/de-en.tgz"
ZIPPED_DATA_PATH = "./.data/de-en.tgz"
DE_DATA_PATH = "./.data/europarl-v7.de-en.de"
EN_DATA_PATH = "./.data/europarl-v7.de-en.en"
