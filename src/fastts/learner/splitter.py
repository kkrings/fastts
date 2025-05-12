from collections.abc import Iterator
from itertools import dropwhile

from torch.nn import Module, Parameter


class BodyHeadSplitter:
    def __init__(self, body: Module) -> None:
        self._body = body

    def head(self, model: Module) -> Iterator[Parameter]:
        body = self._body.parameters()

        def body_has_more_params(_: Parameter) -> bool:
            parameter = next(body, False)
            return parameter is not False

        return dropwhile(body_has_more_params, model.parameters())
