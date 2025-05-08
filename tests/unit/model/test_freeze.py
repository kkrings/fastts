from transformers import DistilBertForSequenceClassification

from fastts.model.freeze import freeze_base_model


def test_freeze_base_model(model: DistilBertForSequenceClassification) -> None:
    with freeze_base_model(model):  # type: ignore
        assert all(not param.requires_grad for param in model.base_model.parameters())  # type: ignore


def test_unfreeze_base_model(model: DistilBertForSequenceClassification) -> None:
    with freeze_base_model(model):  # type: ignore
        ...

    assert all(param.requires_grad for param in model.base_model.parameters())  # type: ignore
