from dataclasses import dataclass

import pytest
import torch
from transformers.modeling_outputs import SequenceClassifierOutput  # type: ignore

from fastai_ttc.callbacks.logits import LogitsToPred


@dataclass
class FakeLearner:
    pred: SequenceClassifierOutput | torch.Tensor


def test_loss_from_model(
    cb: LogitsToPred, learn: FakeLearner, pred: SequenceClassifierOutput
) -> None:
    cb("after_pred")
    assert torch.all(torch.isclose(learn.pred, pred.logits))


@pytest.fixture
def pred() -> SequenceClassifierOutput:
    return SequenceClassifierOutput(logits=torch.tensor([0.1, 0.2, 0.3]))


@pytest.fixture
def learn(pred: SequenceClassifierOutput) -> FakeLearner:
    return FakeLearner(pred)


@pytest.fixture
def cb(learn: FakeLearner) -> LogitsToPred:
    cb = LogitsToPred()
    cb.learn = learn
    return cb
