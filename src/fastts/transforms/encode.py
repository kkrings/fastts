from typing import cast

from fastai.text.all import Transform, typedispatch  # type: ignore
from transformers import BatchEncoding, PreTrainedTokenizerBase  # type: ignore

from fastts.types.textbatch import TextBatch


@typedispatch  # type: ignore
def text_to_text_batch(text: str) -> TextBatch:
    return TextBatch((text,))


class TokenizeText(Transform):  # type: ignore
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, truncation: bool = False
    ) -> None:
        self._tokenizer = tokenizer
        self._truncation = truncation

    def encodes(self, text: TextBatch) -> BatchEncoding:
        batch_encoding = self._tokenizer(
            list(text), padding=True, truncation=self._truncation, return_tensors="pt"
        )

        return cast(BatchEncoding, batch_encoding)
