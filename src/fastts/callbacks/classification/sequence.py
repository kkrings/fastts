from collections.abc import Iterator

from fastai.text.all import Callback  # type: ignore

from fastts.callbacks.logits import LogitsToPred
from fastts.callbacks.loss import LossFromModel
from fastts.callbacks.model import TransformersModel
from fastts.callbacks.targets import TargetsAsLabels


def sequence_classification(loss_from_model: bool = False) -> Iterator[Callback]:
    yield TransformersModel()

    if loss_from_model:
        yield TargetsAsLabels()
        yield LossFromModel()

    yield LogitsToPred()
