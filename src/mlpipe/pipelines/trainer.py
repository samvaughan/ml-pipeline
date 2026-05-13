from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from mlpipe.core.registry import Registry


class ModelTrainingRunner:
    ARTIFACTS_FILENAME = "trainer_artifacts.json"
    CONFIG_FILENAME = "model_trainer.config.json"

    def __init__(
        self,
        trainers: list[Any],
        *,
        evaluators: list[Any],
        config: dict[str, Any] | None = None,
    ):
        self.trainers = trainers
        self.evaluators = evaluators
        self._config = copy.deepcopy(config) if config else None
        self.model_: Any = None
        self.trainer_outputs_: list[Any] = []
        self.last_report_: pd.Series | pd.DataFrame | None = None
        self._log = logging.getLogger(type(self).__name__)
        self._log.info(f"Initialized with {len(self.trainers)} trainer(s) and {len(self.evaluators)} evaluators")

    @classmethod
    def from_cfg(
        cls,
        cfg: dict[str, Any],
        *,
        trainers: Registry,
        evaluators: Registry,
    ) -> ModelTrainingRunner:
        trainer_cfgs = cfg.get("trainers") or []
        built_trainers = [trainers.create(t["name"], **t.get("kwargs", {})) for t in trainer_cfgs]

        evaluator_cfgs = cfg.get("evaluators") or []
        built_evaluators = [evaluators.create(e["name"], **e.get("kwargs", {})) for e in evaluator_cfgs]

        return cls(trainers=built_trainers, evaluators=built_evaluators, config=cfg)

    def to_config(self) -> dict[str, Any]:
        if self._config is not None:
            return copy.deepcopy(self._config)

        cfg: dict[str, Any] = {"trainers": [], "evaluators": []}
        for t in self.trainers:
            cfg["trainers"].append(
                t.to_config() if hasattr(t, "to_config") else {"name": type(t).__name__, "kwargs": {}}
            )
        for e in self.evaluators:
            cfg["evaluators"].append(
                e.to_config() if hasattr(e, "to_config") else {"name": type(e).__name__, "kwargs": {}}
            )
        return cfg

    def fit(self, X_train: Any, y_train: Any, **kwargs) -> ModelTrainingRunner:
        self.trainer_outputs_ = []
        current = None
        for trainer in self.trainers:
            self._log.info(f"Running trainer: {type(trainer).__name__}")
            out = trainer.fit(X=X_train, y=y_train, base_artifact=current, **kwargs)
            self.trainer_outputs_.append(out)
            if out is not None:
                current = out
        self.model_ = current
        self._log.info(f"Training complete. model_ type = {type(self.model_).__name__}")
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        if self.model_ is None:
            raise RuntimeError("Runner not fitted. Call fit() first.")
        last_trainer = self.trainers[-1] if self.trainers else None
        if last_trainer is not None and callable(getattr(last_trainer, "predict", None)):
            return last_trainer.predict(X, **kwargs)
        raise AttributeError("No predict() available on last trainer.")

    def evaluate(self, *, y_true: Any, y_pred: Any, label: str | None = None, **kwargs) -> Any:
        if not self.evaluators:
            raise RuntimeError("No evaluators configured.")
        for e in self.evaluators:
            self._log.info(f"Evaluating using {type(e).__name__}")
            out = e.evaluate(y_true, y_pred, label=label, **kwargs)
            self.last_report_ = out
        return out

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config(output_dir)

        manifest: dict[str, Any] = {"trainers": []}
        for i, t in enumerate(self.trainers):
            t_dir = output_dir / f"trainer_{i}_{type(t).__name__}"
            t_dir.mkdir(exist_ok=True)
            if callable(getattr(t, "save", None)):
                t.save(t_dir)
            manifest["trainers"].append({"index": i, "name": type(t).__name__, "path": t_dir.name})

        with open(output_dir / self.ARTIFACTS_FILENAME, "w") as f:
            json.dump(manifest, f, indent=2)
        joblib.dump(self.model_, output_dir / "final_model.joblib")

    @classmethod
    def load(cls, input_dir: Path, *, trainers: Registry, evaluators: Registry) -> ModelTrainingRunner:
        input_dir = Path(input_dir)

        cfg_path = input_dir / cls.CONFIG_FILENAME
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config: {cfg_path}")
        with open(cfg_path) as f:
            cfg = json.load(f)

        runner = cls.from_cfg(cfg, trainers=trainers, evaluators=evaluators)

        manifest_path = input_dir / cls.ARTIFACTS_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing artifact manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        entries = manifest.get("trainers", [])

        if len(entries) != len(runner.trainers):
            raise ValueError(
                f"Trainer count mismatch: config has {len(runner.trainers)}, manifest has {len(entries)}."
            )

        for entry in entries:
            i = int(entry["index"])
            trainer = runner.trainers[i]
            t_dir = input_dir / entry["path"]
            if not t_dir.exists():
                raise FileNotFoundError(f"Missing trainer artifact dir: {t_dir}")
            cls_load = getattr(type(trainer), "load", None)
            loaded = cls_load(t_dir) if callable(cls_load) else None
            if loaded is not None:
                runner.trainers[i] = loaded

        final_path = input_dir / "final_model.joblib"
        runner.model_ = joblib.load(final_path) if final_path.exists() else runner.trainers[-1]
        return runner

    def _save_config(self, output_dir: Path) -> None:
        with open(output_dir / self.CONFIG_FILENAME, "w") as f:
            json.dump(self.to_config(), f, indent=2)
