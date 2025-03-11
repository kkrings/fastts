from dataclasses import dataclass
from typing import Generic, TypeGuard, TypeVar

import torch
from transformers.modeling_outputs import SequenceClassifierOutput  # type: ignore

from fastai_ttc.callbacks.models.distilbert import DistilBERTArgs

_X = TypeVar("_X")


@dataclass
class Learner(Generic[_X]):
    x: _X
    xb: tuple[_X] | DistilBERTArgs
    pred: SequenceClassifierOutput | torch.FloatTensor
    loss_grad: torch.FloatTensor
    loss: torch.FloatTensor


def is_distilbert_args(xb: tuple[_X] | DistilBERTArgs) -> TypeGuard[DistilBERTArgs]:
    return len(xb) == 2
