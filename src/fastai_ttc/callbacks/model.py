from collections.abc import Callable
from typing import TypeVar

import torch
from fastai.text.all import Callback  # type: ignore
from transformers import BatchEncoding  # type: ignore

_Pred = TypeVar("_Pred")


class TTCModel(Callback):  # type: ignore
    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None

    def before_batch(self) -> None:
        self._model = self.model
        self.learn.model = self._wrap_model(self.model)

    def after_pred(self) -> None:
        self.learn.model = self._model

    @staticmethod
    def _wrap_model(model: Callable[..., _Pred]) -> Callable[[BatchEncoding], _Pred]:
        def wrapper(x: BatchEncoding) -> _Pred:
            return model(**x)

        return wrapper
