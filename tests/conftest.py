from typing import cast

import pandas as pd
import pytest
import torch
from fastai.text.all import (  # type: ignore
    ColReader,
    ColSplitter,
    DataBlock,
    DataLoaders,
    Learner,
    RegressionBlock,
)
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from fastai_ttc.callbacks.logits import LogitsToPred
from fastai_ttc.callbacks.loss import LossFromModel
from fastai_ttc.callbacks.model import TransformersModel
from fastai_ttc.callbacks.targets import TargetsAsLabels
from fastai_ttc.transforms.block import TTCBlock


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased",
        cache_dir=tmp_path_factory.mktemp("tokenizer"),
    )


@pytest.fixture(scope="session")
def model(tmp_path_factory: pytest.TempPathFactory) -> torch.nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        cache_dir=tmp_path_factory.mktemp("model"),
        num_labels=1,
    )

    return cast(torch.nn.Module, model)


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


@pytest.fixture
def learn(dls: DataLoaders, model: torch.nn.Module) -> Learner:
    return Learner(
        dls,
        model,
        cbs=[TransformersModel, TargetsAsLabels, LossFromModel, LogitsToPred],
    )
