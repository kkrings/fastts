from dataclasses import dataclass

import pytest
import torch
from transformers.modeling_outputs import SequenceClassifierOutput  # type: ignore

from fastai_ttc.callbacks.loss import LossFromModel


@dataclass
class FakeLearner:
    pred: SequenceClassifierOutput
    loss_grad: torch.Tensor
    loss: torch.Tensor


def test_loss_from_model(
    cb: LossFromModel, learn: FakeLearner, pred: SequenceClassifierOutput
) -> None:
    cb("after_pred")
    cb("after_loss")
    assert torch.all(torch.isclose(learn.loss_grad, pred.loss))
    assert torch.all(torch.isclose(learn.loss, pred.loss))


@pytest.fixture
def pred() -> SequenceClassifierOutput:
    return SequenceClassifierOutput(loss=torch.tensor([0.5]))


@pytest.fixture
def loss() -> torch.Tensor:
    return torch.tensor([1.0])


@pytest.fixture
def learn(loss: torch.Tensor, pred: SequenceClassifierOutput) -> FakeLearner:
    return FakeLearner(pred, loss_grad=loss, loss=loss)


@pytest.fixture
def cb(learn: FakeLearner) -> LossFromModel:
    cb = LossFromModel()
    cb.learn = learn
    return cb
