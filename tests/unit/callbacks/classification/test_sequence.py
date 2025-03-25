import pytest
from fastai.text.all import Callback  # type: ignore

from fastts.callbacks.classification.sequence import sequence_classification
from fastts.callbacks.logits import LogitsToPred
from fastts.callbacks.loss import LossFromModel
from fastts.callbacks.model import TransformersModel
from fastts.callbacks.targets import TargetsAsLabels


@pytest.mark.parametrize(
    ("loss_from_model", "expected_cbs"),
    (
        (False, (TransformersModel, LogitsToPred)),
        (True, (TransformersModel, TargetsAsLabels, LossFromModel, LogitsToPred)),
    ),
)
def test_sequence_classification(
    loss_from_model: bool, expected_cbs: tuple[type[Callback], ...]
) -> None:
    cbs = tuple(type(cb) for cb in sequence_classification(loss_from_model))
    assert cbs == expected_cbs
