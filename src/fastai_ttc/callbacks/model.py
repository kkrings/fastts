from fastai.text.all import Callback  # type: ignore


class TTCModel(Callback):  # type: ignore
    def before_batch(self) -> None:
        x = {key: self.x[..., i] for i, key in enumerate(self.x.keys_along_last_dim)}
        self.learn.xb = (x["input_ids"], x["attention_mask"], x["token_type_ids"])

    def after_pred(self) -> None:
        self.learn.pred = self.pred.logits
