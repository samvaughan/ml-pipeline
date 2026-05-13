import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import joblib
import pandas as pd

from ltv.core.registries import DATA_SELECTORS, TRANSFORMERS
from ltv.plugins.selectors.data_selector import RowSelector
from ltv.utils.logging_utils import LoggerMixin


class DataFinaliser(LoggerMixin):
    def __init__(
        self,
        transformers: List[Any],
        output_column_spec: Dict[str, Any],
        train_test_split_column: str,
        bad_client_ids: List[int] | None = None,
        selectors_by_split: Optional[Mapping[Any, RowSelector]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.transformers = transformers
        self.output_column_spec = output_column_spec
        self.train_test_split_column = train_test_split_column

        self.bad_client_ids = bad_client_ids or []

        # NEW: per-split selectors (e.g. {"train": ..., "val": ..., "test": ...})
        self.selectors_by_split = dict(selectors_by_split) if selectors_by_split else {}

        self._config: Optional[Dict[str, Any]] = None
        if config is not None:
            self._config = copy.deepcopy(config)

        self.log(
            f"Initialized DataFinaliser with {len(self.transformers)} transformers. "
            f"bad_client_ids={self.bad_client_ids}. "
            f"selectors={list(self.selectors_by_split.keys())}"
        )

    # -------- Construction from YAML config --------
    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "DataFinaliser":
        """
        cfg is the `DataFinaliser.config` dict from your YAML
        """
        # `or []`/`or {}` (not `, []`/`, {}`) so an explicit null section in
        # YAML — e.g. `transformers:` followed by just a comment — is treated
        # the same as a missing key. `.get(k, default)` only returns the
        # default when the key is absent, not when the value is None.
        transformer_cfgs = cfg.get("transformers") or []
        selectors_cfg = cfg.get("selectors") or {}

        output_column_spec = cfg.get("output_column_spec", {})
        train_test_split_column = cfg.get("train_test_split_column", "")
        bad_client_ids = cfg.get("bad_client_ids", None)

        if transformer_cfgs:
            transformers = [
                TRANSFORMERS.create(t["name"], **t.get("kwargs", {}))
                for t in transformer_cfgs
            ]
        else:
            transformers = []

        selectors_by_split = {
            split_value: DATA_SELECTORS.create(sel["name"], **sel.get("kwargs", {}))
            for split_value, sel in selectors_cfg.items()
        }

        return cls(
            transformers=transformers,
            output_column_spec=output_column_spec,
            train_test_split_column=train_test_split_column,
            bad_client_ids=bad_client_ids,
            selectors_by_split=selectors_by_split,
            config=cfg,
        )

    # -------- Config Saving --------
    def to_config(self) -> Dict[str, Any]:
        """
        Return a config dict representing this pipeline's setup.

        Shape matches what `from_cfg` expects:
          {
            "transformers": [...]
          }

        If the pipeline was built with `from_cfg`, this will be exactly the same
        (modulo deep copy). If not, it will try a best-effort reconstruction
        from the instance types.
        """
        if self._config is not None:
            return copy.deepcopy(self._config)

        # Fallback: reconstruct minimal config from instances
        cfg: Dict[str, Any] = {"cleaners": [], "transformers": []}

        for t in self.transformers:
            if hasattr(t, "to_config") and callable(t.to_config):
                cfg["transformers"].append(t.to_config())
            else:
                cfg["transformers"].append(
                    {
                        "name": type(t).__name__,
                        "kwargs": {},
                    }
                )

        return cfg

    def _inject_context(self) -> None:
        # If our plugins have a 'set_context' method, this will call it for them
        # Needed by the 'standardise' method, which needs to know the output feature columns
        ctx = {"output_column_spec": self.output_column_spec}
        for t in self.transformers:
            if hasattr(t, "set_context"):
                t.set_context(**ctx)

    def split_datasets(self, data: pd.DataFrame):
        base = self.output_column_spec["feature_columns"]
        requested = base

        missing = [c for c in requested if c not in data.columns]
        if missing:
            raise KeyError(
                f"Missing {len(missing)} feature columns at split_datasets time. "
                f"First few: {missing[:10]}"
            )

        target_column = self.output_column_spec["target_column"]
        metadata_columns = self.output_column_spec["metadata_columns"]

        return {
            "features": data[requested],
            "target": data.loc[:, target_column],
            "metadata": data[metadata_columns],
        }

    # -------- Core API --------
    def fit(self, df: pd.DataFrame) -> "DataFinaliser":
        """
        Fit transformers

        """
        self._inject_context()
        data = df.copy()
        # 2) Transformers in sequence
        for t in self.transformers:
            self.log(f"Fitting transformer: {type(t).__name__}")
            # Allow fit() to either:
            #   - mutate data in-place, or
            #   - return a new dataframe (for "fit-time feature materialisation")
            maybe = t.fit(data)
            if isinstance(maybe, pd.DataFrame):
                data = maybe

        return self

    def transform(
        self, df: pd.DataFrame, training: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply cleaners + transformers to raw df.

        Returns the final object from the last transformer
        """
        self._inject_context()
        data = df.copy()
        # 2) Transformers in sequence
        for t in self.transformers:
            self.log(f"Applying transformer: {type(t).__name__}")
            data = t.transform(data, training=training)

        return data

    def filter_good_clients(self, data: pd.DataFrame) -> pd.DataFrame:
        self.log(f"Removing the following client IDs: {self.bad_client_ids}")

        return data.loc[(~data.client_id.isin(self.bad_client_ids))]

    def build_dataset(self, data: pd.DataFrame, selector: RowSelector) -> dict:
        selected = selector.select(data)
        return self.split_datasets(selected)

    def finalise_by_split(
        self,
        data: pd.DataFrame,
        selectors_by_split: Dict[Any, RowSelector],
    ) -> Dict[Any, Dict[str, pd.DataFrame]]:
        """
        Apply a different RowSelector per split defined by self.train_test_split_column.

        If `train_test_split_column` is falsy, the data is treated as a single
        split and emitted under one key (taken from selectors_by_split if
        present, otherwise "train"). A missing or empty selector means "no row
        selection"; all rows are passed through.

        Returns:
            {
            split_value: {"features": X, "target": y, "metadata": meta},
            ...
            }
        """
        # Single-split mode: no train/val/test column → emit one split
        if not self.train_test_split_column:
            split_value = next(iter(selectors_by_split), "train")
            selector = selectors_by_split.get(split_value)
            self.log(
                f"No train_test_split_column configured; emitting single "
                f"split={split_value!r}"
                + (
                    f" with selector={type(selector).__name__}"
                    if selector is not None
                    else " (no selector)"
                )
            )
            selected = selector.select(data) if selector is not None else data
            return {split_value: self.split_datasets(selected)}

        if self.train_test_split_column not in data.columns:
            raise KeyError(
                f"Split column '{self.train_test_split_column}' not found in transformed data."
            )

        # output dict to save the results into
        out: Dict[Any, Dict[str, pd.DataFrame]] = {}

        for split_value, split_df in data.groupby(
            self.train_test_split_column, sort=False
        ):
            if split_value not in selectors_by_split:
                raise KeyError(
                    f"No selector provided for split_value={split_value!r}. "
                    f"Available selectors: {list(selectors_by_split.keys())}"
                )

            selector = selectors_by_split[split_value]
            self.log(
                f"Building dataset for split={split_value!r} using selector={type(selector).__name__}"
            )

            out[split_value] = self.build_dataset(split_df, selector=selector)

        return out

    def fit_transform(self, df: pd.DataFrame, training: bool = True) -> Any:
        """
        Convenience for training: fit on df then transform it.
        """
        self.fit(df)

        data = self.transform(df, training=training)
        data = self.filter_good_clients(data)

        return self.finalise_by_split(data, selectors_by_split=self.selectors_by_split)

    # -------- Persistence --------
    def save(self, output_dir: Path) -> None:
        """
        Save the entire pipeline (cleaners + transformers + learned state)
        in one joblib artifact.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_dir / "data_finaliser.joblib")
        self.save_config(output_dir)
        self.log(f"Saved DataFinaliser to {output_dir}")

    @classmethod
    def load(cls, input_dir: Path) -> "DataFinaliser":
        """
        Load a previously saved DataFinaliser.
        """
        input_dir = Path(input_dir)
        pipeline: "DataFinaliser" = joblib.load(input_dir / "data_finaliser.joblib")
        return pipeline

    def save_config(
        self, output_dir: Path, filename: str = "data_finaliser.config.json"
    ) -> None:
        """
        Convenience method to save the pipeline config to disk as JSON.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.to_config()
        with open(output_dir / filename, "w") as f:
            json.dump(cfg, f, indent=2)
        self.log(f"Saved DataFinaliser config to {output_dir / filename}")
