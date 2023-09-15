"""
This module holds all the data related classes and functions.
"""

__all__ = ["preprocess"]


import io
from collections import Counter
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
from torchtext.vocab import Vocab, vocab

from transformer.config import BATCH_SIZE, TEST_TRAIN_VAL_SPLIT
from transformer.constants import (
    DATA_DOWNLOAD_SOURCE,
    DE_DATA_PATH,
    EN_DATA_PATH,
    ZIPPED_DATA_PATH,
)


class GermanEnglishDataset(Dataset):
    """
    German-English-Dataset
    """

    def __init__(self):
        """
        Initialization method.
        """
        self._prepare_data()
        self.de_data, self.en_data = self._open_data()
        self.de_tokenizer, self.en_tokenizer = self._get_tokenizers()
        self.de_vocab = self._build_vocabulary(DE_DATA_PATH, self.de_tokenizer)
        self.en_vocab = self._build_vocabulary(EN_DATA_PATH, self.en_tokenizer)
        self._tokenize_data()

    def _prepare_data(self) -> None:
        """
        Prepare the data by downloading and extracting it.
        """
        download_from_url(url=DATA_DOWNLOAD_SOURCE)
        extract_archive(from_path=ZIPPED_DATA_PATH)

    def _open_data(self) -> tuple:
        """
        Open the data from the given paths.
        :return: tuple of German and English data
        """
        return io.open(DE_DATA_PATH, encoding="utf8"), io.open(
            EN_DATA_PATH, encoding="utf8"
        )

    def _get_tokenizers(self) -> tuple:
        """
        Get the tokenizers for the data.
        :return: tuple of tokenizers, first for German, second for English
        """
        return get_tokenizer("spacy", language="de_core_news_sm"), get_tokenizer(
            "spacy", language="en_core_web_sm"
        )

    def _build_vocabulary(self, filepath: str, tokenizer: partial) -> Vocab:
        """
        Builds a vocabulary from a text file (found at filepath)
        using the given tokenizer.
        :param filepath: path of the file
        :param tokenizer: tokenizer to use
        :return: vocabulary
        """
        counter = Counter()
        with open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        return vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])

    def _tokenize_data(self) -> None:
        """
        Load the data into tensors and tokenzie them on the way.
        Use the raw data, tokenizers and vocabularies of this class.
        """
        data = []
        for (raw_de, raw_en) in zip(self.de_data, self.en_data):
            de_tensor_ = torch.tensor(
                [self.de_vocab[token] for token in self.de_tokenizer(raw_de)],
                dtype=torch.long,
            )
            en_tensor_ = torch.tensor(
                [self.en_vocab[token] for token in self.en_tokenizer(raw_en)],
                dtype=torch.long,
            )
            data.append((de_tensor_, en_tensor_))
        self.data = data

    def __len__(self) -> int:
        """
        Return the length of the data.
        :return: length of the data
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        """
        Use the given index and return the data at this index.
        :param index: index
        :return: tuple of German and English data
        """
        return self.data[index]


def generate_batch(data_batch, vocabulary: Vocab) -> tuple:
    """
    Collate function to generate a batch from the given data.
    This makes sure that the data is padded correctly.
    :param data_batch: batch of data
    :param vocabulary: vocabulary to use
    :return: tuple of German and English padded batch
    """
    PAD_IDX = vocabulary["<pad>"]
    BOS_IDX = vocabulary["<bos>"]
    EOS_IDX = vocabulary["<eos>"]

    de_batch, en_batch = [], []

    for (de_item, en_item) in data_batch:
        de_batch.append(
            torch.cat(
                [torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0
            )
        )
        en_batch.append(
            torch.cat(
                [torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0
            )
        )

    return pad_sequence(de_batch, padding_value=PAD_IDX), pad_sequence(
        en_batch, padding_value=PAD_IDX
    )


def preprocess():
    """
    Main preprocess function to preprocess all the data.
    This includes initializing the german to englis dataset,
    splitting it into training, testing and validation and
    returning the corresponding dataloaders.
    """
    dataset = GermanEnglishDataset()

    test_data, train_data, val_data = random_split(dataset, TEST_TRAIN_VAL_SPLIT)

    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: generate_batch(batch, dataset.de_vocab),
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: generate_batch(batch, dataset.de_vocab),
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: generate_batch(batch, dataset.de_vocab),
    )
    return train_dataloader, test_dataloader, val_dataloader
