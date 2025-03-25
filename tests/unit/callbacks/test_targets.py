from dataclasses import dataclass

import pytest
import torch
from transformers import BatchEncoding  # type: ignore

from fastts.callbacks.targets import TargetsAsLabels


@dataclass
class FakeLearner:
    xb: tuple[BatchEncoding]
    yb: tuple[torch.Tensor]
    x: BatchEncoding
    y: torch.Tensor


def test_targets_as_labels(
    cb: TargetsAsLabels, learn: FakeLearner, y: torch.Tensor
) -> None:
    cb("before_batch")
    assert torch.all(torch.isclose(learn.xb[0]["labels"], y))


@pytest.fixture
def x() -> BatchEncoding:
    return BatchEncoding(data={"input_ids": torch.tensor([5, 9, 13])})


@pytest.fixture
def y() -> torch.Tensor:
    return torch.tensor([0, 1, 2])


@pytest.fixture
def learn(x: BatchEncoding, y: torch.Tensor) -> FakeLearner:
    return FakeLearner(xb=(x,), yb=(y,), x=x, y=y)


@pytest.fixture
def cb(learn: FakeLearner) -> TargetsAsLabels:
    cb = TargetsAsLabels()
    cb.learn = learn
    return cb
