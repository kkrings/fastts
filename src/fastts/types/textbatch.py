from collections.abc import Iterable, Iterator, Sequence
from typing import overload


class TextBatch(Sequence[str]):
    def __init__(self, text: Iterable[str | Iterable[str]]) -> None:
        self._batch = tuple(self._flatten(text))

    def __len__(self) -> int:
        return len(self._batch)

    @overload
    def __getitem__(self, select: int) -> str: ...

    @overload
    def __getitem__(self, select: slice) -> Sequence[str]: ...

    def __getitem__(self, select: int | slice) -> str | Sequence[str]:
        return self._batch[select]

    @staticmethod
    def _flatten(text: Iterable[str | Iterable[str]]) -> Iterator[str]:
        for s in text:
            if isinstance(s, str):
                yield s
            else:
                yield from s
