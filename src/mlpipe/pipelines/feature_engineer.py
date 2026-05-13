import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from ltv.core.registries import CLEANERS, TRANSFORMERS
from ltv.utils.logging_utils import LoggerMixin


class FeatureEngineer(LoggerMixin):
    """
    End-to-end pipeline:
      - apply cleaners to raw DataFrame
      - apply feature engineers
      - return final dataframe
    """

    def __init__(
        self,
        cleaners: List[Any],
        transformers: List[Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.cleaners = cleaners
        self.transformers = transformers

        # Store the original config (cleaners + transformers) so we can round-trip it
        # Shape is expected to be: {"cleaners": [...], "transformers": [...]}
        self._config: Optional[Dict[str, Any]] = None
        if config is not None:
            # Make a defensive copy so we don't accidentally mutate caller's dict
            self._config = copy.deepcopy(config)

        self.log(
            f"Initialized FeatureEngineer with "
            f"{len(self.cleaners)} cleaners and {len(self.transformers)} transformers"
        )

    # -------- Construction from YAML config --------
    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "FeatureEngineer":
        """
        cfg is the `FeatureEngineer.config` dict from your YAML
        """
        # `or []` (not `, []`) so an explicit null / empty-section in YAML is
        # treated the same as a missing key — `transformers:` with a comment
        # below parses to None, not [].
        cleaner_cfgs = cfg.get("cleaners") or []
        transformer_cfgs = cfg.get("transformers") or []

        cleaners = [
            CLEANERS.create(c["name"], **c.get("kwargs", {})) for c in cleaner_cfgs
        ]

        transformers = [
            TRANSFORMERS.create(t["name"], **t.get("kwargs", {}))
            for t in transformer_cfgs
        ]

        return cls(cleaners=cleaners, transformers=transformers, config=cfg)

    # -------- Config Saving --------
    def to_config(self) -> Dict[str, Any]:
        """
        Return a config dict representing this pipeline's setup.

        Shape matches what `from_cfg` expects:
          {
            "cleaners": [...],
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

        for c in self.cleaners:
            # If cleaner has its own to_config(), defer to that
            if hasattr(c, "to_config") and callable(c.to_config):
                cfg["cleaners"].append(c.to_config())
            else:
                cfg["cleaners"].append(
                    {
                        "name": type(c).__name__,
                        "kwargs": {},  # we don't know original kwargs here
                    }
                )

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

    # -------- Core API --------
    def fit(self, df: pd.DataFrame, **kwargs) -> "FeatureEngineer":
        """
        Fit any stateful transformers on cleaned data.

        """
        self.log("Fitting MarginFeaturePipeline")

        context = kwargs.get("context", {})

        # 1) Apply cleaners to a copy of the raw df
        clean_df = df.copy()
        for c in self.cleaners:
            self.log(f"Applying cleaner: {type(c).__name__}")
            self._validate_context_for_fit(c, context)
            clean_df = c.transform(clean_df, **kwargs)

        # 2) Fit each transformer on the cleaned DataFrame
        # NOTE issue here- data for fit is never updated to the output of previous transformer
        # So if a transformer depends on previous transformer's output, it won't work
        # Will need to fix if I ever have such a transformer, where its fit method depends on the outputs of a previous transformer's output
        # This is fine for now since all of my transformers can fit on the cleaned DataFrame directly
        data_for_fit: Any = clean_df
        for t in self.transformers:
            self.log(f"Fitting transformer: {type(t).__name__}")
            self._validate_context_for_fit(t, context)
            # margin_feature_engineer.fit will use DataFrame
            # EntropyCalculator.fit is a no-op and can accept anything
            t.fit(data_for_fit, **kwargs)

            # critical: evolve the data so the next transformer can fit on derived columns
            # Note that this means we end up transforming twice, but it's worth it to avoid overcomplicating things
            # We do have a flag to skip this for slow transformers
            if getattr(t, "run_during_fit", True):
                data_for_fit = t.transform(data_for_fit, training=True, **kwargs)
            else:
                self.log(f"Not running transform method for {type(t).__name__}")

            # # optionally run transform to produce inputs for downstream fitters
            # if hasattr(t, "transform"):
            #     data_for_fit = t.transform(data_for_fit, training=True)

        return self

    def transform(self, df: pd.DataFrame, training: bool = False, **kwargs) -> Any:
        """
        Apply cleaners + transformers to raw df.

        Returns the final object from the last transformer
        """
        self.log(f"Transform called (training={training})")

        # context = kwargs.get("context", {})

        # 1) Cleaners on DataFrame
        data: Any = df.copy()
        for c in self.cleaners:
            self.log(f"Applying cleaner: {type(c).__name__}")
            data = c.transform(data, **kwargs)

        # 2) Transformers in sequence
        for t in self.transformers:
            self.log(f"Applying transformer: {type(t).__name__}")
            data = t.transform(data, training=training, **kwargs)

        return data

    def fit_transform(self, df: pd.DataFrame, training: bool = True, **kwargs) -> Any:
        """
        Convenience for training: fit on df then transform it.
        """
        self.fit(df, **kwargs)
        return self.transform(df, training=training, **kwargs)

    # -------- Persistence --------
    def save(self, output_dir: Path) -> None:
        """
        Save the entire pipeline (cleaners + transformers + learned state)
        in one joblib artifact.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_dir / "feature_pipeline.joblib")
        self.save_config(output_dir)
        self.log(f"Saved FeatureEngineer to {output_dir}")

    @classmethod
    def load(cls, input_dir: Path) -> "FeatureEngineer":
        """
        Load a previously saved FeatureEngineer.
        """
        input_dir = Path(input_dir)
        pipeline: "FeatureEngineer" = joblib.load(input_dir / "feature_pipeline.joblib")
        return pipeline

    def save_config(
        self, output_dir: Path, filename: str = "feature_pipeline.config.json"
    ) -> None:
        """
        Convenience method to save the pipeline config to disk as JSON.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.to_config()
        with open(output_dir / filename, "w") as f:
            json.dump(cfg, f, indent=2)
        self.log(f"Saved FeatureEngineer config to {output_dir / filename}")

    def _validate_context_for_fit(self, transformer, context):
        required = getattr(transformer, "requires_context_keys_for_fit", [])
        if not required:
            return

        if context is None:
            raise ValueError(
                f"{type(transformer).__name__} requires context keys "
                f"{required}, but no context was provided."
            )

        missing = [k for k in required if k not in context]
        if missing:
            raise ValueError(
                f"{type(transformer).__name__} requires context keys {missing}, "
                f"but they were not found in context."
            )
