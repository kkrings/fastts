from collections.abc import Iterator

from torch.nn import Module, Parameter


class BodyHeadSplitter:
    def __init__(self, body: Module) -> None:
        self._body = body

    def __call__(self, model: Module) -> list[list[Parameter]]:
        parameters = model.parameters()
        return [list(self.body(parameters)), list(parameters)]

    def body(self, parameters: Iterator[Parameter]) -> Iterator[Parameter]:
        return (p for _, p in zip(self._body.parameters(), parameters))
