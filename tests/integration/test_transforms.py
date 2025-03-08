import pandas as pd
from fastai.text.all import DataLoaders  # type: ignore


def test_transforms(dls: DataLoaders, df: pd.DataFrame) -> None:
    batch = dls.decode_batch(dls.one_batch())
    row = df.loc[0]
    assert batch == [(row["input"].lower(), row["target"])]
