from fastai.text.all import L, TensorCategory, TensorText, Transform  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore
from typing_extensions import Unpack

from fastai_ttc.transforms.types import TransformCallKwargs


class TTCTokenizer(Transform):  # type: ignore
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, truncation: bool = False
    ) -> None:
        self._tokenizer = tokenizer
        self._truncation = truncation

    def __call__(
        self, batch: list[tuple[str, TensorCategory]], **_: Unpack[TransformCallKwargs]
    ) -> tuple[tuple[TensorText, TensorCategory], ...]:
        xb = self._tokenizer(
            [x for x, _ in batch],
            padding=True,
            truncation=self._truncation,
            return_tensors="pt",
        )

        xbt = TensorText(
            L(xb.values()).stack(dim=-1), keys_along_last_dim=tuple(xb.keys())
        )

        return tuple(zip((x for x in xbt), (y for _, y in batch)))
