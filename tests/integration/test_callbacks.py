from dataclasses import dataclass
from typing import Annotated, Generic, TypeGuard, TypeVar

import pandas as pd
import pytest
import torch
from fastai.text.all import DataLoaders  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from fastai_ttc.callbacks.model import TTCModel

X = TypeVar("X")

ModelArgs = tuple[
    Annotated[torch.Tensor, "input_ids"],
    Annotated[torch.Tensor, "attention_mask"],
]


@dataclass
class Learner(Generic[X]):
    xb: tuple[X] | ModelArgs
    x: X


def is_model_args(xb: tuple[X] | ModelArgs) -> TypeGuard[ModelArgs]:
    return len(xb) == 2


def test_callbacks(dls: DataLoaders, xb: ModelArgs) -> None:
    x, _ = dls.one_batch()
    learn = Learner(xb=(x,), x=x)
    cb = TTCModel()
    cb.learn = learn
    cb("before_batch")
    assert is_model_args(learn.xb)
    assert torch.all(learn.xb[0] == xb[0])
    assert torch.all(learn.xb[1] == xb[1])


@pytest.fixture
def xb(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> ModelArgs:
    x = tokenizer(text=df["input"][0], return_tensors="pt")
    return tuple(x[key] for key in ("input_ids", "attention_mask"))
