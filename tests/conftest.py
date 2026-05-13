import pandas as pd
from mlpipe.base.transformer import BaseTransformer
from mlpipe import Registry
import pytest


class AddColumnTransformer(BaseTransformer):
    def __init__(self, column: str, value: int = 1):
        self.column = column
        self.value = value

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, training=False, **kwargs):
        df = df.copy()
        df[self.column] = self.value
        return df


@pytest.fixture
def transformer_registry():
    reg = Registry("transformer")
    reg.register("add_col")(AddColumnTransformer)
    return reg


@pytest.fixture
def simple_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
