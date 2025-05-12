from fastai.text.all import Learner  # type: ignore
from transformers import DistilBertForSequenceClassification


def test_freeze_body(
    learn: Learner, model: DistilBertForSequenceClassification
) -> None:
    learn.freeze()
    assert all(not p.requires_grad for p in model.base_model.parameters())  # type: ignore


def test_freeze_head(
    learn: Learner, model: DistilBertForSequenceClassification
) -> None:
    learn.freeze()

    assert all(
        p.requires_grad
        for name, p in model.named_parameters()  # type: ignore
        if not name.startswith(model.base_model_prefix)  # type: ignore
    )


def test_unfreeze(learn: Learner, model: DistilBertForSequenceClassification) -> None:
    learn.freeze()
    learn.unfreeze()
    assert all(p.requires_grad for p in model.parameters())  # type: ignore
