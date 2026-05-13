from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from mlpipe.core.registry import Registry


class DataFinaliser:
    def __init__(
        self,
        transformers: list[Any],
        output_column_spec: dict[str, Any],
        train_test_split_column: str = "",
        excluded_ids: list[Any] | None = None,
        id_column: str | None = None,
        selectors_by_split: dict[Any, Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.transformers = transformers
        self.output_column_spec = output_column_spec
        self.train_test_split_column = train_test_split_column
        self.excluded_ids = excluded_ids or []
        self.id_column = id_column
        self.selectors_by_split = dict(selectors_by_split) if selectors_by_split else {}
        self._config = copy.deepcopy(config) if config is not None else None
        self._log = logging.getLogger(type(self).__name__)
        self._log.info(
            f"Initialized DataFinaliser with {len(self.transformers)} transformers. "
            f"excluded_ids={len(self.excluded_ids)} rows. "
            f"selectors={list(self.selectors_by_split.keys())}"
        )

    # -------- Construction from YAML config --------
    @classmethod
    def from_cfg(
        cls,
        cfg: dict[str, Any],
        *,
        transformers: Registry,
        selectors: Registry,
    ) -> DataFinaliser:
        transformer_cfgs = cfg.get("transformers") or []
        selectors_cfg = cfg.get("selectors") or {}

        built_transformers = [
            transformers.create(t["name"], **t.get("kwargs", {}))
            for t in transformer_cfgs
        ]
        selectors_by_split = {
            split_value: selectors.create(sel["name"], **sel.get("kwargs", {}))
            for split_value, sel in selectors_cfg.items()
        }

        return cls(
            transformers=built_transformers,
            output_column_spec=cfg.get("output_column_spec", {}),
            train_test_split_column=cfg.get("train_test_split_column", ""),
            excluded_ids=cfg.get("excluded_ids"),
            id_column=cfg.get("id_column"),
            selectors_by_split=selectors_by_split,
            config=cfg,
        )

    # -------- Config Saving --------
    def to_config(self) -> dict[str, Any]:
        if self._config is not None:
            return copy.deepcopy(self._config)

        cfg: dict[str, Any] = {"transformers": []}
        for t in self.transformers:
            cfg["transformers"].append(
                t.to_config() if hasattr(t, "to_config") else {"name": type(t).__name__, "kwargs": {}}
            )
        return cfg

    def _inject_context(self) -> None:
        ctx = {"output_column_spec": self.output_column_spec}
        for t in self.transformers:
            if hasattr(t, "set_context"):
                t.set_context(**ctx)

    def split_datasets(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        feature_columns = self.output_column_spec["feature_columns"]
        missing = [c for c in feature_columns if c not in data.columns]
        if missing:
            raise KeyError(
                f"Missing {len(missing)} feature columns at split_datasets time. "
                f"First few: {missing[:10]}"
            )

        target_column = self.output_column_spec["target_column"]
        metadata_columns = self.output_column_spec["metadata_columns"]

        return {
            "features": data[feature_columns],
            "target": data.loc[:, target_column],
            "metadata": data[metadata_columns],
        }

    # -------- Core API --------
    def fit(self, df: pd.DataFrame) -> DataFinaliser:
        self._inject_context()
        data = df.copy()
        for t in self.transformers:
            self._log.info(f"Fitting transformer: {type(t).__name__}")
            maybe = t.fit(data)
            if isinstance(maybe, pd.DataFrame):
                data = maybe
        return self

    def transform(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        self._inject_context()
        data = df.copy()
        for t in self.transformers:
            self._log.info(f"Applying transformer: {type(t).__name__}")
            data = t.transform(data, training=training)
        return data

    def _filter_excluded_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.id_column or not self.excluded_ids:
            return data
        self._log.info(
            f"Removing {len(self.excluded_ids)} excluded IDs from column '{self.id_column}'"
        )
        return data.loc[~data[self.id_column].isin(self.excluded_ids)]

    def build_dataset(self, data: pd.DataFrame, selector: Any) -> dict[str, pd.DataFrame]:
        selected = selector.select(data)
        return self.split_datasets(selected)

    def finalise_by_split(
        self,
        data: pd.DataFrame,
        selectors_by_split: dict[Any, Any],
    ) -> dict[Any, dict[str, pd.DataFrame]]:
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
        if not self.train_test_split_column:
            split_value = next(iter(selectors_by_split), "train")
            selector = selectors_by_split.get(split_value)
            self._log.info(
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

        out: dict[Any, dict[str, pd.DataFrame]] = {}
        for split_value, split_df in data.groupby(self.train_test_split_column, sort=False):
            if split_value not in selectors_by_split:
                raise KeyError(
                    f"No selector provided for split_value={split_value!r}. "
                    f"Available selectors: {list(selectors_by_split.keys())}"
                )
            selector = selectors_by_split[split_value]
            self._log.info(
                f"Building dataset for split={split_value!r} using selector={type(selector).__name__}"
            )
            out[split_value] = self.build_dataset(split_df, selector=selector)

        return out

    def fit_transform(self, df: pd.DataFrame, training: bool = True) -> dict[Any, dict[str, pd.DataFrame]]:
        self.fit(df)
        data = self.transform(df, training=training)
        data = self._filter_excluded_rows(data)
        return self.finalise_by_split(data, selectors_by_split=self.selectors_by_split)

    # -------- Persistence --------
    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_dir / "data_finaliser.joblib")
        self.save_config(output_dir)
        self._log.info(f"Saved DataFinaliser to {output_dir}")

    @classmethod
    def load(cls, input_dir: Path) -> DataFinaliser:
        return joblib.load(Path(input_dir) / "data_finaliser.joblib")

    def save_config(
        self, output_dir: Path, filename: str = "data_finaliser.config.json"
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.to_config()
        with open(output_dir / filename, "w") as f:
            json.dump(cfg, f, indent=2)
        self._log.info(f"Saved DataFinaliser config to {output_dir / filename}")
