import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase  # type: ignore


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased",
        cache_dir=tmp_path_factory.mktemp("tokenizer"),
    )
