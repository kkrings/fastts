from fastai.text.all import Learner  # type: ignore
from transformers import DistilBertForSequenceClassification

from fastts.learner.splitter import BodyHeadSplitter


def test_freeze_body(
    learn: Learner, model: DistilBertForSequenceClassification
) -> None:
    learn.freeze()
    assert all(not p.requires_grad for p in model.base_model.parameters())  # type: ignore


def test_freeze_head(
    learn: Learner, model: DistilBertForSequenceClassification
) -> None:
    learn.freeze()
    splitter = BodyHeadSplitter(body=model.base_model)  # type: ignore
    assert all(p.requires_grad for p in splitter.head(model))  # type: ignore


def test_unfreeze(learn: Learner, model: DistilBertForSequenceClassification) -> None:
    learn.freeze()
    learn.unfreeze()
    assert all(p.requires_grad for p in model.parameters())  # type: ignore
