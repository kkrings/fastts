from fastai.text.all import TensorText, TitledStr, Transform  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore


class TTCDecoder(Transform):  # type: ignore
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def decodes(self, x: TensorText) -> TitledStr:
        return TitledStr(self._tokenizer.decode(x[..., 0], skip_special_tokens=True))
