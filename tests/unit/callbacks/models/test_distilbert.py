import pandas as pd
import pytest
import torch
from fastai.text.all import DataLoaders  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore
from transformers.modeling_outputs import SequenceClassifierOutput  # type: ignore

from fastai_ttc.callbacks.model import TransformersModel
from fastai_ttc.callbacks.models.distilbert import DistilBERTArgs
from tests.utils.learner import Learner, is_distilbert_args


@pytest.mark.parametrize(
    ["loss", "model_loss", "use_loss_from_model"],
    [(torch.tensor([1.0]), torch.tensor([0.5]), True)],
)
def test_ttc_model_before_batch(
    cb: TransformersModel,
    dls: DataLoaders,
    loss: torch.FloatTensor,
    model_output: SequenceClassifierOutput,
    xb: DistilBERTArgs,
) -> None:
    x, _ = dls.one_batch()
    learn = Learner(x, xb=(x,), pred=model_output, loss_grad=loss, loss=loss)
    cb.learn = learn
    cb("before_batch")
    assert is_distilbert_args(learn.xb)
    assert torch.all(learn.xb[0] == xb[0])
    assert torch.all(learn.xb[1] == xb[1])


@pytest.mark.parametrize(
    ["loss", "model_loss", "use_loss_from_model"],
    [(torch.tensor([1.0]), torch.tensor([0.5]), True)],
)
def test_ttc_model_after_pred(
    cb: TransformersModel,
    dls: DataLoaders,
    loss: torch.FloatTensor,
    model_output: SequenceClassifierOutput,
) -> None:
    x, _ = dls.one_batch()
    learn = Learner(x, xb=(x,), pred=model_output, loss_grad=loss, loss=loss)
    cb.learn = learn
    cb("after_pred")
    assert isinstance(learn.pred, torch.FloatTensor)
    assert torch.all(torch.isclose(learn.pred, model_output.logits))


@pytest.mark.parametrize(
    ["loss", "model_loss", "use_loss_from_model", "expected_loss"],
    [
        (torch.tensor([1.0]), torch.tensor([0.5]), True, torch.tensor([0.5])),
        (torch.tensor([1.0]), torch.tensor([0.5]), False, torch.tensor([1.0])),
        (torch.tensor([1.0]), None, True, torch.tensor([1.0])),
    ],
)
def test_ttc_model_after_loss(
    cb: TransformersModel,
    dls: DataLoaders,
    loss: torch.FloatTensor,
    model_output: SequenceClassifierOutput,
    expected_loss: torch.FloatTensor,
) -> None:
    x, _ = dls.one_batch()
    learn = Learner(x, xb=(x,), pred=model_output, loss_grad=loss, loss=loss)
    cb.learn = learn
    cb("after_pred")
    cb("after_loss")
    assert torch.all(torch.isclose(learn.loss_grad, expected_loss))
    assert torch.all(torch.isclose(learn.loss, expected_loss))


@pytest.fixture
def model_output(model_loss: torch.FloatTensor | None) -> SequenceClassifierOutput:
    return SequenceClassifierOutput(loss=model_loss, logits=torch.tensor([0.1, 0.2]))


@pytest.fixture
def cb(use_loss_from_model: bool) -> TransformersModel:
    return TransformersModel(use_loss_from_model)


@pytest.fixture
def xb(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> DistilBERTArgs:
    x = tokenizer(text=df["input"][0], return_tensors="pt")
    return tuple(x[key] for key in ("input_ids", "attention_mask"))
