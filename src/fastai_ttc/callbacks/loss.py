import torch
from fastai.text.all import Callback  # type: ignore


class LossFromModel(Callback):  # type: ignore
    def __init__(self) -> None:
        self._loss_from_model: torch.Tensor | None = None

    def after_pred(self) -> None:
        self._loss_from_model = self.pred.loss

    def after_loss(self) -> None:
        if self._loss_from_model is None:
            return

        self.learn.loss_grad = self._loss_from_model
        self.learn.loss = self._loss_from_model.clone()
