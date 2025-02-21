from fastai.text.all import SortedDL, TransformBlock  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore

from fastai_ttc.transforms.decoder import TTCDecoder
from fastai_ttc.transforms.tokenizer import TTCTokenizer


class TTCBlock(TransformBlock):  # type: ignore
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, truncation: bool = False
    ) -> None:
        super().__init__(
            type_tfms=[TTCDecoder(tokenizer)],
            dl_type=SortedDL,
            dls_kwargs={"before_batch": TTCTokenizer(tokenizer, truncation)},
        )
