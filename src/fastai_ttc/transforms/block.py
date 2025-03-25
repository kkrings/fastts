from fastai.text.all import SortedDL, Transform, TransformBlock  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from fastai_ttc.transforms.decode import DecodeInputIds, input_ids_to_tensor_text
from fastai_ttc.transforms.encode import TokenizeText, text_to_text_batch


class TransformersTextBlock(TransformBlock):  # type: ignore
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, truncation: bool = False
    ) -> None:
        super().__init__(
            type_tfms=[
                Transform(text_to_text_batch),
                DecodeInputIds(tokenizer),
            ],
            batch_tfms=[
                TokenizeText(tokenizer, truncation),
                Transform(dec=input_ids_to_tensor_text),
            ],
            dl_type=SortedDL,
        )
