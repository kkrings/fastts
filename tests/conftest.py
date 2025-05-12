from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from fastai.text.all import (  # type: ignore
    ColReader,
    ColSplitter,
    DataBlock,
    DataLoaders,
    Learner,
    RegressionBlock,
)
from transformers import (  # type: ignore
    AutoTokenizer,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerBase,
)

from fastts.callbacks.classification.sequence import sequence_classification
from fastts.transforms.block import TransformersTextBlock


@pytest.fixture(scope="session")
def cache_root_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


@pytest.fixture
def tokenizer(cache_root_dir: Path) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased",
        cache_dir=cache_root_dir / "tokenizer",
    )

    return cast(PreTrainedTokenizerBase, tokenizer)


@pytest.fixture
def model(cache_root_dir: Path) -> DistilBertForSequenceClassification:
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        cache_dir=cache_root_dir / "model",
        num_labels=1,
    )

    return cast(DistilBertForSequenceClassification, model)


@pytest.fixture
def dblock(tokenizer: PreTrainedTokenizerBase) -> DataBlock:
    return DataBlock(
        blocks=(TransformersTextBlock(tokenizer), RegressionBlock),
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
    return dblock.dataloaders(df, bs=1, num_workers=0)


@pytest.fixture
def learn(dls: DataLoaders, model: DistilBertForSequenceClassification) -> Learner:
    return Learner(dls, model, cbs=list(sequence_classification(loss_from_model=True)))
