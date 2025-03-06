from dataclasses import dataclass

import pytest
import torch
from transformers.modeling_outputs import SequenceClassifierOutput  # type: ignore

from fastai_ttc.callbacks.model import TTCModel


@dataclass
class Learner:
    pred: SequenceClassifierOutput | torch.FloatTensor
    loss_grad: torch.FloatTensor
    loss: torch.FloatTensor


@pytest.mark.parametrize(
    ["loss", "model_loss", "use_loss_from_model"],
    [(torch.tensor([1.0]), torch.tensor([0.5]), True)],
)
def test_ttc_model_after_pred(
    cb: TTCModel, learn: Learner, model_output: SequenceClassifierOutput
) -> None:
    cb("after_pred")
    assert isinstance(learn.pred, torch.FloatTensor)
    assert torch.all(torch.isclose(learn.pred, model_output.logits))


@pytest.mark.parametrize(
    ["loss", "model_loss", "use_loss_from_model", "expected_loss"],
    [
        (torch.tensor([1.0]), torch.tensor([0.5]), True, torch.tensor([0.5])),
        (torch.tensor([1.0]), torch.tensor([0.5]), False, torch.tensor([1.0])),
        (torch.tensor([1.0]), None, True, torch.tensor([1.0])),
    ],
)
def test_ttc_model_after_loss(
    cb: TTCModel, learn: Learner, expected_loss: torch.FloatTensor
) -> None:
    cb("after_pred")
    cb("after_loss")
    assert torch.all(torch.isclose(learn.loss_grad, expected_loss))
    assert torch.all(torch.isclose(learn.loss, expected_loss))


@pytest.fixture
def model_output(model_loss: torch.FloatTensor | None) -> SequenceClassifierOutput:
    return SequenceClassifierOutput(loss=model_loss, logits=torch.tensor([0.1, 0.2]))


@pytest.fixture
def learn(loss: torch.FloatTensor, model_output: SequenceClassifierOutput) -> Learner:
    return Learner(pred=model_output, loss_grad=loss, loss=loss)


@pytest.fixture
def cb(use_loss_from_model: bool, learn: Learner) -> TTCModel:
    cb = TTCModel(use_loss_from_model)
    cb.learn = learn
    return cb
