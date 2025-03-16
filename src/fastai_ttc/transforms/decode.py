from fastai.text.all import (  # type: ignore
    TensorText,
    TitledStr,
    Transform,
    typedispatch,
)
from transformers import BatchEncoding, PreTrainedTokenizerBase  # type: ignore


@typedispatch  # type: ignore
def input_ids_to_tensor_text(tokens: BatchEncoding) -> TensorText:
    return TensorText(tokens.input_ids)


class DecodeInputIds(Transform):  # type: ignore
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def decodes(self, input_ids: TensorText) -> TitledStr:
        return TitledStr(self._tokenizer.decode(token_ids=input_ids))
