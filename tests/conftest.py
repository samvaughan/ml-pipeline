import pandas as pd
from mlpipe.base.transformer import BaseTransformer
from mlpipe import Registry
import pytest
from unittest.mock import MagicMock
from mlpipe import FeatureEngineer


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
def simple_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture
def transformer_registry():
    reg = Registry("transformer")
    reg.register("add_col")(AddColumnTransformer)
    return reg


@pytest.fixture
def feature_engineer_cfg():
    # To be used with the above transformer_registry
    return {
        "cleaners": [],
        "transformers": [{"name": "add_col", "kwargs": {"column": "x", "value": 5}}],
    }


@pytest.fixture
def cleaner_registry():
    return Registry("cleaner")


@pytest.fixture
def mock_transformers(simple_df):
    t1 = MagicMock(spec=BaseTransformer)
    t2 = MagicMock(spec=BaseTransformer)
    t1.transform.return_value = simple_df
    t2.transform.return_value = simple_df
    return (t1, t2)


@pytest.fixture
def mock_transformers_no_run_during_fit(mock_transformers):
    t1, t2 = mock_transformers
    t1.run_during_fit = False
    t2.run_during_fit = False
    return (t1, t2)


@pytest.fixture
def mock_transformers_needing_context(mock_transformers):
    t1, t2 = mock_transformers
    t1.requires_context_keys_for_fit = ["test_key"]
    t2.requires_context_keys_for_fit = ["test_key_2"]
    return (t1, t2)


@pytest.fixture
def feature_engineer(feature_engineer_cfg, transformer_registry, cleaner_registry):
    return FeatureEngineer.from_cfg(
        feature_engineer_cfg,
        transformers=transformer_registry,
        cleaners=cleaner_registry,
    )
