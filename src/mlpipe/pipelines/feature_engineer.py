from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from mlpipe.base.transformer import BaseTransformer
from mlpipe.core.registry import Registry


class FeatureEngineer:
    def __init__(
        self,
        cleaners: list[Any],
        transformers: list[Any],
        config: dict[str, Any] | None = None,
    ):
        self.cleaners = cleaners
        self.transformers = transformers
        self._config = copy.deepcopy(config) if config is not None else None
        self._log = logging.getLogger(type(self).__name__)
        self._log.info(
            f"Initialized with {len(self.cleaners)} cleaners and {len(self.transformers)} transformers"
        )

    @classmethod
    def from_cfg(
        cls,
        cfg: dict[str, Any],
        *,
        transformers: Registry,
        cleaners: Registry,
    ) -> FeatureEngineer:
        cleaner_cfgs = cfg.get("cleaners") or []
        transformer_cfgs = cfg.get("transformers") or []

        built_cleaners = [
            cleaners.create(c["name"], **c.get("kwargs", {})) for c in cleaner_cfgs
        ]
        built_transformers = [
            transformers.create(t["name"], **t.get("kwargs", {}))
            for t in transformer_cfgs
        ]

        return cls(cleaners=built_cleaners, transformers=built_transformers, config=cfg)

    def to_config(self) -> dict[str, Any]:
        # Fallback reconstruction is best-effort; prefer building via from_cfg for round-trip safety.
        if self._config is not None:
            return copy.deepcopy(self._config)

        cfg: dict[str, Any] = {"cleaners": [], "transformers": []}
        for c in self.cleaners:
            cfg["cleaners"].append(
                c.to_config()
                if type(c).to_config is not BaseTransformer.to_config
                else {"name": type(c).__name__, "kwargs": {}}
            )
        for t in self.transformers:
            cfg["transformers"].append(
                t.to_config()
                if type(t).to_config is not BaseTransformer.to_config
                else {"name": type(t).__name__, "kwargs": {}}
            )
        return cfg

    def fit(self, df: pd.DataFrame, **kwargs) -> FeatureEngineer:
        context = kwargs.get("context", {})

        clean_df = df.copy()
        for c in self.cleaners:
            self._log.info(f"Applying cleaner: {type(c).__name__}")
            self._validate_context(c, context)
            clean_df = c.transform(clean_df, **kwargs)

        data_for_fit: Any = clean_df
        for t in self.transformers:
            self._log.info(f"Fitting transformer: {type(t).__name__}")
            self._validate_context(t, context)
            t.fit(data_for_fit, **kwargs)

            if getattr(t, "run_during_fit", True):
                data_for_fit = t.transform(data_for_fit, training=True, **kwargs)
            else:
                self._log.info(f"Skipping transform during fit for {type(t).__name__}")

        return self

    def transform(self, df: pd.DataFrame, training: bool = False, **kwargs) -> Any:
        data: Any = df.copy()
        for c in self.cleaners:
            data = c.transform(data, **kwargs)
        for t in self.transformers:
            self._log.info(f"Applying transformer: {type(t).__name__}")
            data = t.transform(data, training=training, **kwargs)
        return data

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> Any:
        self.fit(df, **kwargs)
        return self.transform(df, training=True, **kwargs)

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_dir / "feature_pipeline.joblib")
        self._save_config(output_dir)
        self._log.info(f"Saved FeatureEngineer to {output_dir}")

    @classmethod
    def load(cls, input_dir: Path) -> FeatureEngineer:
        return joblib.load(Path(input_dir) / "feature_pipeline.joblib")

    def _save_config(
        self, output_dir: Path, filename: str = "feature_pipeline.config.json"
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / filename, "w") as f:
            json.dump(self.to_config(), f, indent=2)

    def _validate_context(self, obj: Any, context: dict) -> None:
        required = getattr(obj, "requires_context_keys_for_fit", [])
        if not required:
            return
        missing = [k for k in required if k not in context]
        if missing:
            raise ValueError(
                f"{type(obj).__name__} requires context keys {missing} but they were not provided."
            )
