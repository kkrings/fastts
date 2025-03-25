from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from transformers import BatchEncoding  # type: ignore

from fastai_ttc.callbacks.model import TTCModel


@dataclass
class FakeLearner:
    model: Callable[..., Any]


def test_transformers_model_before_batch(
    cb: TTCModel,
    learn: FakeLearner,
    xb: tuple[BatchEncoding],
    model: MagicMock,
    x: BatchEncoding,
) -> None:
    cb("before_batch")
    learn.model(*xb)
    model.assert_called_once_with(input_ids=x["input_ids"])


def test_transformers_model_after_pred(
    cb: TTCModel, learn: FakeLearner, model: MagicMock
) -> None:
    cb("before_batch")
    cb("after_pred")
    assert learn.model is model


@pytest.fixture
def model(mocker: MockerFixture) -> MagicMock:
    return mocker.stub(name="model")


@pytest.fixture
def learn(model: MagicMock) -> FakeLearner:
    return FakeLearner(model)


@pytest.fixture
def cb(learn: FakeLearner) -> TTCModel:
    cb = TTCModel()
    cb.learn = learn
    return cb


@pytest.fixture
def x() -> BatchEncoding:
    return BatchEncoding(data={"input_ids": "some input ids"})


@pytest.fixture
def xb(x: BatchEncoding) -> tuple[BatchEncoding]:
    return (x,)
