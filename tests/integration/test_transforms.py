import pandas as pd
import pytest
from fastai.text.all import (  # type: ignore
    ColReader,
    ColSplitter,
    DataBlock,
    DataLoaders,
    RegressionBlock,
)
from transformers import PreTrainedTokenizerBase  # type: ignore

from fastai_ttc.transforms.block import TTCBlock


def test_transforms(dls: DataLoaders, df: pd.DataFrame) -> None:
    batch = dls.decode_batch(dls.one_batch())
    assert batch == [(df["input"][0].lower(), df["target"][0])]


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "input": [
                "I am part of the training set",
                "I am part of the validation set",
            ],
            "target": [0.5, 1.0],
            "validation": [False, True],
        }
    )


@pytest.fixture
def dblock(tokenizer: PreTrainedTokenizerBase) -> DataBlock:
    return DataBlock(
        blocks=(TTCBlock(tokenizer), RegressionBlock),
        get_x=ColReader("input"),
        get_y=ColReader("target"),
        splitter=ColSplitter("validation"),
    )


@pytest.fixture
def dls(dblock: DataBlock, df: pd.DataFrame) -> DataLoaders:
    return dblock.dataloaders(df, bs=1)
