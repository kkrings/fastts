import torch
from fastai.text.all import Callback  # type: ignore


class TTCModel(Callback):  # type: ignore
    def __init__(self, use_loss_from_model: bool = False) -> None:
        self._use_loss_from_model = use_loss_from_model
        self._loss_from_model: torch.FloatTensor | None = None

    def before_batch(self) -> None:
        self.learn.xb = (self.x["input_ids"], self.x["attention_mask"])

    def after_pred(self) -> None:
        self._loss_from_model = self.pred.loss
        self.learn.pred = self.pred.logits

    def after_loss(self) -> None:
        if not self._use_loss_from_model or self._loss_from_model is None:
            return

        self.learn.loss_grad = self._loss_from_model
        self.learn.loss = self.loss_grad.clone()
