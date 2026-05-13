from mlpipe import FeatureEngineer
from mlpipe.base.transformer import BaseTransformer
import pandas as pd
import pytest
from unittest.mock import ANY, MagicMock


def test_config_round_trip(feature_engineer, feature_engineer_cfg):
    assert feature_engineer.to_config() == feature_engineer_cfg


def test_calling_fit_transform_raises_no_errors(feature_engineer, simple_df):

    feature_engineer.fit_transform(df=simple_df)


def test_fit_calls_each_transformer(mock_transformers, simple_df):
    t1, t2 = mock_transformers

    fe = FeatureEngineer(cleaners=[], transformers=[t1, t2])
    fe.fit(simple_df)

    t1.fit.assert_called_once()
    t2.fit.assert_called_once()


def test_transform_calls_each_transformer(mock_transformers, simple_df):
    t1, t2 = mock_transformers

    fe = FeatureEngineer(cleaners=[], transformers=[t1, t2])
    fe.transform(simple_df)

    t1.transform.assert_called_once()
    t2.transform.assert_called_once()

    t1.fit.assert_not_called()
    t2.fit.assert_not_called()


def test_fit_transform_calls_each_transformer(mock_transformers, simple_df):
    t1, t2 = mock_transformers

    fe = FeatureEngineer(cleaners=[], transformers=[t1, t2])
    fe.fit_transform(simple_df)

    t1.fit.assert_called_once()
    t2.fit.assert_called_once()

    assert t1.transform.call_count == 2
    assert t2.transform.call_count == 2


def test_fit_transform_matches_fit_then_transform(feature_engineer, simple_df):
    fit_then_transform = feature_engineer.fit(simple_df).transform(simple_df)
    fit_transform = feature_engineer.fit_transform(simple_df)

    pd.testing.assert_frame_equal(fit_then_transform, fit_transform)


def test_fit_with_run_during_fit_false(mock_transformers_no_run_during_fit, simple_df):
    t1, t2 = mock_transformers_no_run_during_fit

    fe = FeatureEngineer(cleaners=[], transformers=[t1, t2])
    fe.fit(simple_df)

    t1.fit.assert_called_once()
    t2.fit.assert_called_once()

    t1.transform.assert_not_called()
    t2.transform.assert_not_called()


def test_fit_with_transformers_requiring_context(
    mock_transformers_needing_context, simple_df
):
    t1, t2 = mock_transformers_needing_context

    fe = FeatureEngineer(cleaners=[], transformers=[t1, t2])
    context = {"test_key": "value1", "test_key_2": "value2"}
    fe.fit(simple_df, context=context)

    t1.fit.assert_called_once_with(ANY, context=context)
    t2.fit.assert_called_once_with(ANY, context=context)


def test_fit_with_transformers_requiring_context_raises_error_when_key_missing(
    mock_transformers_needing_context, simple_df
):
    t1, t2 = mock_transformers_needing_context

    fe = FeatureEngineer(cleaners=[], transformers=[t1, t2])
    context = {}
    with pytest.raises(ValueError, match="requires context keys"):
        fe.fit(simple_df, context=context)


def test_save_load_round_trip(
    feature_engineer, feature_engineer_cfg, simple_df, tmp_path
):
    feature_engineer.fit(simple_df)
    feature_engineer.save(tmp_path)

    loaded = FeatureEngineer.load(tmp_path)

    assert loaded.to_config() == feature_engineer_cfg
    pd.testing.assert_frame_equal(
        loaded.transform(simple_df),
        feature_engineer.transform(simple_df),
    )
    assert (tmp_path / "feature_pipeline.config.json").exists()


def test_transform_chains_output_between_transformers(simple_df):
    chained_df = simple_df.assign(new_col=99)
    t1 = MagicMock(spec=BaseTransformer)
    t2 = MagicMock(spec=BaseTransformer)
    t1.transform.return_value = chained_df
    t2.transform.return_value = chained_df

    fe = FeatureEngineer(cleaners=[], transformers=[t1, t2])
    fe.transform(simple_df)

    t2.transform.assert_called_once_with(chained_df, training=False)


def test_cleaners_run_before_transformers(simple_df):
    cleaned_df = simple_df.assign(cleaned=True)
    cleaner = MagicMock(spec=BaseTransformer)
    cleaner.transform.return_value = cleaned_df
    transformer = MagicMock(spec=BaseTransformer)
    transformer.transform.return_value = cleaned_df

    fe = FeatureEngineer(cleaners=[cleaner], transformers=[transformer])
    fe.fit(simple_df)

    first_arg = transformer.fit.call_args[0][0]
    pd.testing.assert_frame_equal(first_arg, cleaned_df)


def test_from_cfg_raises_for_unknown_transformer(
    transformer_registry, cleaner_registry
):
    cfg = {"cleaners": [], "transformers": [{"name": "nonexistent", "kwargs": {}}]}
    with pytest.raises(KeyError):
        FeatureEngineer.from_cfg(
            cfg, transformers=transformer_registry, cleaners=cleaner_registry
        )


def test_to_config_fallback_when_built_directly():
    class MinimalTransformer(BaseTransformer):
        def fit(self, df, **kwargs):
            return self

        def transform(self, df, training=False, **kwargs):
            return df

    fe = FeatureEngineer(cleaners=[], transformers=[MinimalTransformer()])
    assert fe.to_config() == {
        "cleaners": [],
        "transformers": [{"name": "MinimalTransformer", "kwargs": {}}],
    }
