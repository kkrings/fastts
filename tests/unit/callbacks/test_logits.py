from dataclasses import dataclass

import pytest
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

from fastts.callbacks.logits import LogitsToPred


@dataclass
class FakeLearner:
    pred: SequenceClassifierOutput | torch.Tensor


def test_logits_to_pred(
    cb: LogitsToPred, learn: FakeLearner, logits: torch.Tensor
) -> None:
    cb("after_pred")
    assert isinstance(learn.pred, torch.Tensor)
    assert torch.all(torch.isclose(learn.pred, logits))


@pytest.fixture
def logits() -> torch.Tensor:
    return torch.tensor([0.1, 0.2, 0.3])


@pytest.fixture
def pred(logits: torch.FloatTensor) -> SequenceClassifierOutput:
    return SequenceClassifierOutput(logits=logits)


@pytest.fixture
def learn(pred: SequenceClassifierOutput) -> FakeLearner:
    return FakeLearner(pred)


@pytest.fixture
def cb(learn: FakeLearner) -> LogitsToPred:
    cb = LogitsToPred()
    cb.learn = learn
    return cb
