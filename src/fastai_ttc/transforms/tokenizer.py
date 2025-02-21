from fastai.text.all import L, TensorCategory, TensorText, Transform  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore
from typing_extensions import Unpack

from fastai_ttc.transforms.types import TransformCallKwargs


class TTCTokenizer(Transform):  # type: ignore
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def __call__(
        self, batch: list[tuple[str, TensorCategory]], **_: Unpack[TransformCallKwargs]
    ) -> tuple[tuple[TensorText, TensorCategory], ...]:
        xb = self._tokenizer([x for x, _ in batch], padding=True, return_tensors="pt")

        return tuple(
            zip(
                (x for x in TensorText(L(xb.values()).stack(dim=-1))),
                (y for _, y in batch),
            )
        )
