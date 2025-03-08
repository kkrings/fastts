import pandas as pd
import pytest
from fastai.text.all import (  # type: ignore
    ColReader,
    ColSplitter,
    DataBlock,
    DataLoaders,
    RegressionBlock,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase  # type: ignore

from fastai_ttc.transforms.block import TTCBlock


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased",
        cache_dir=tmp_path_factory.mktemp("tokenizer"),
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
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "input": [
                "I am part of the training set",
                "I am part of the validation set",
            ],
            "target": [
                0.5,
                1.0,
            ],
            "validation": [
                False,
                True,
            ],
        }
    )


@pytest.fixture
def dls(dblock: DataBlock, df: pd.DataFrame) -> DataLoaders:
    return dblock.dataloaders(df, bs=1)
