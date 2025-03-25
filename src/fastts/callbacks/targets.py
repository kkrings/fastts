from fastai.text.all import Callback  # type: ignore
from transformers import BatchEncoding  # type: ignore


class TargetsAsLabels(Callback):  # type: ignore
    def before_batch(self) -> None:
        x = BatchEncoding(data=self.x)
        x["labels"] = self.y
        self.learn.xb = (x,)
