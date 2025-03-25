from fastai.text.all import Callback  # type: ignore


class LogitsToPred(Callback):  # type: ignore
    def after_pred(self) -> None:
        self.learn.pred = self.pred.logits
