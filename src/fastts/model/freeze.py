from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch
from transformers import PreTrainedModel


def require_grad(parameters: Iterable[torch.nn.Parameter], value: bool) -> None:
    for param in parameters:
        param.requires_grad = value


def freeze(parameters: Iterable[torch.nn.Parameter]) -> None:
    require_grad(parameters, False)


def unfreeze(parameters: Iterable[torch.nn.Parameter]) -> None:
    require_grad(parameters, True)


@contextmanager
def freeze_base_model(model: PreTrainedModel) -> Iterator[None]:
    try:
        freeze(model.base_model.parameters())  # type: ignore
        yield
    finally:
        unfreeze(model.base_model.parameters())  # type: ignore
