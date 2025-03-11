import pandas as pd
import pytest
from fastai.text.all import DataLoaders  # type: ignore
from transformers import PreTrainedTokenizerBase  # type: ignore


def test_transforms(dls: DataLoaders, text: str, target: float) -> None:
    batch = dls.decode_batch(dls.one_batch())
    assert batch == [(text, target)]


@pytest.fixture
def text(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> str:
    return str(tokenizer.decode(token_ids=tokenizer(text=df["input"][0])["input_ids"]))


@pytest.fixture
def target(df: pd.DataFrame) -> float:
    return float(df["target"][0])
