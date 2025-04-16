from dataclasses import dataclass

import pytest
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

from fastts.callbacks.loss import LossFromModel


@dataclass
class FakeLearner:
    pred: SequenceClassifierOutput
    loss_grad: torch.Tensor
    loss: torch.Tensor


def test_loss_from_model(
    cb: LossFromModel, learn: FakeLearner, loss_from_model: torch.Tensor
) -> None:
    cb("after_pred")
    cb("after_loss")
    assert torch.all(torch.isclose(learn.loss_grad, loss_from_model))
    assert torch.all(torch.isclose(learn.loss, loss_from_model))


@pytest.fixture
def loss() -> torch.Tensor:
    return torch.tensor([1.0])


@pytest.fixture
def loss_from_model() -> torch.Tensor:
    return torch.tensor([0.5])


@pytest.fixture
def pred(loss_from_model: torch.FloatTensor) -> SequenceClassifierOutput:
    return SequenceClassifierOutput(loss=loss_from_model)


@pytest.fixture
def learn(loss: torch.Tensor, pred: SequenceClassifierOutput) -> FakeLearner:
    return FakeLearner(pred, loss_grad=loss, loss=loss)


@pytest.fixture
def cb(learn: FakeLearner) -> LossFromModel:
    cb = LossFromModel()
    cb.learn = learn
    return cb
