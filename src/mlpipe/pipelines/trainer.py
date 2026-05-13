from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from ltv.core.registries import EVALUATORS, TRAINERS
from ltv.utils.logging_utils import LoggerMixin


class ModelTrainingRunner(LoggerMixin):
    """
    End-to-end training pipeline:
      - run trainer plugins in sequence
      - produce a trained model/artifact
      - evaluate/plot/save via a Report (e.g. PyMCRegressionReport)

    Conventions (recommended):
      - Each trainer implements:
          fit(data, training=True) -> Any
        and returns either:
          - a trained model/artifact, OR
          - a container that includes a model, OR
          - None (trainer mutates internal state)

      - The final "trained artifact" will be stored in self.model_.
    """

    ARTIFACTS_FILENAME = "trainer_artifacts.json"
    CONFIG_FILENAME = "model_trainer.config.json"

    def __init__(
        self,
        trainers: List[Any],
        *,
        evaluators: List[Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.trainers = trainers
        self.evaluators = evaluators

        self._config: Optional[Dict[str, Any]] = (
            copy.deepcopy(config) if config else None
        )

        self.model_: Any = None
        self.trainer_outputs_: List[Any] = []
        self.last_report_: Optional[pd.Series | pd.DataFrame] = None

        self.log(
            f"Initialized ModelTrainingRunner with {len(self.trainers)} trainer(s) "
            f"and {len(self.evaluators)} evaluators"
        )

    # -------- Construction from YAML config --------
    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "ModelTrainingRunner":
        """
        Expected shape:
          {
            "trainers": [
              {"name": "pymc_regression_trainer", "kwargs": {...}},
              {"name": "posterior_calibrator", "kwargs": {...}}
            ],
            "report": { ... }  # optional; often constructed elsewhere
          }
        """
        # `or []` not `, []`: an explicit null YAML section (`evaluators:` with
        # a comment below) parses to None, which .get(..., []) does NOT catch.
        trainer_cfgs = cfg.get("trainers") or []
        trainers = [
            TRAINERS.create(t["name"], **t.get("kwargs", {})) for t in trainer_cfgs
        ]

        evaluators_cfgs = cfg.get("evaluators") or []
        evaluators = [
            EVALUATORS.create(t["name"], **t.get("kwargs", {})) for t in evaluators_cfgs
        ]

        return cls(trainers=trainers, evaluators=evaluators, config=cfg)

    def to_config(self) -> Dict[str, Any]:
        if self._config is not None:
            return copy.deepcopy(self._config)

        cfg: Dict[str, Any] = {"trainers": []}
        for t in self.trainers:
            if hasattr(t, "to_config") and callable(t.to_config):
                cfg["trainers"].append(t.to_config())
            else:
                cfg["trainers"].append({"name": type(t).__name__, "kwargs": {}})

        for e in self.evaluators:
            if hasattr(e, "to_config") and callable(e.to_config):
                cfg["evaluators"].append(e.to_config())
            else:
                cfg["evaluators"].append({"name": type(e).__name__, "kwargs": {}})

        return cfg

    # -------- Core API --------
    def fit(self, X_train: Any, y_train: Any, **kwargs) -> "ModelTrainingRunner":
        """
        Run training plugins in sequence. Stores final artifact in self.model_.
        """
        self.log("Fitting ModelTrainingRunner")

        self.trainer_outputs_ = []

        current = None
        for trainer in self.trainers:
            self.log(f"Running trainer: {type(trainer).__name__}")
            out = trainer.fit(X=X_train, y=y_train, base_artifact=current, **kwargs)

            self.trainer_outputs_.append(out)

            # Carry forward if trainer returns something
            if out is not None:
                current = out

        self.model_ = current
        self.log(f"Training complete. model_ type = {type(self.model_).__name__}")
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        """
        Optional convenience. Assumes the trained artifact has a predict method,
        or the last trainer exposes predict.
        """
        if self.model_ is None:
            raise RuntimeError("Runner not fitted. Call fit() first.")

        # # Common options:
        # if hasattr(self.model_, "predict") and callable(self.model_.predict):
        #     return self.model_.predict(X, **kwargs)

        last_trainer = self.trainers[-1] if self.trainers else None
        if (
            last_trainer is not None
            and hasattr(last_trainer, "predict")
            and callable(last_trainer.predict)
        ):
            return last_trainer.predict(X, **kwargs)

        raise AttributeError("No predict() available on model_ or last trainer.")

    def evaluate(
        self, *, y_true: Any, y_pred: Any, label: Optional[str] = None, **kwargs
    ) -> pd.Series:
        """
        Calls report.evaluate(y_true, y_pred, label=...).
        """
        if len(self.evaluators) == 0:
            raise RuntimeError("No Evaluators configured; cannot evaluate().")

        for e in self.evaluators:
            self.log(f"Evaluating (label={label}) using {type(e).__name__}")
            report_out = e.evaluate(y_true, y_pred, label=label, **kwargs)
            self.last_report_ = report_out
        return report_out

    def plot(
        self,
        *,
        y_true: Any,
        y_pred: Any,
        output_dir: Optional[Path] = None,
        filename: str = "evaluation_plot.png",
        **plot_kwargs,
    ):
        """
        Calls report.plot(y_true, y_pred, **plot_kwargs). If output_dir is provided,
        saves the figure there.

        PyMCRegressionReport.plot returns a matplotlib Figure.
        """
        if len(self.evaluators) == 0:
            raise RuntimeError("No Evaluators configured; cannot evaluate().")

        for e in self.evaluators:
            self.log(f"Plotting using {type(e).__name__}")
            fig, ax = e.plot(y_true, y_pred, **plot_kwargs)

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / filename
                fig.savefig(path, dpi=150, bbox_inches="tight")
                self.log(f"Saved plot to {path}")

        return fig, ax

    # -------- Persistence --------
    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Save config
        self.save_config(output_dir)

        # 2) Save trainer artifacts + manifest
        manifest: Dict[str, Any] = {"trainers": []}

        for i, t in enumerate(self.trainers):
            name = type(t).__name__
            t_dir = output_dir / f"trainer_{i}_{name}"
            t_dir.mkdir(exist_ok=True)

            entry = {
                "index": i,
                "name": name,
                "path": t_dir.name,  # store relative folder name
                "has_save": bool(hasattr(t, "save") and callable(getattr(t, "save"))),
                "has_load": bool(hasattr(t, "load") and callable(getattr(t, "load"))),
                "has_class_load": bool(
                    hasattr(type(t), "load") and callable(getattr(type(t), "load"))
                ),
            }

            if entry["has_save"]:
                t.save(t_dir)

            manifest["trainers"].append(entry)

        with open(output_dir / self.ARTIFACTS_FILENAME, "w") as f:
            json.dump(manifest, f, indent=2)

        # Optional: persist where "model_" should be sourced from
        # (defaults to last trainer with model_ or last trainer)
        manifest_model = {"model_source": "last_trainer_model_attr_or_trainer"}
        with open(output_dir / "model_pointer.json", "w") as f:
            json.dump(manifest_model, f, indent=2)

        joblib.dump(self.model_, output_dir / "final_model.joblib")

    @classmethod
    def load(cls, input_dir: Path) -> "ModelTrainingRunner":
        """
        Robust loader that:
          1) Reconstructs runner + trainers from config
          2) Loads each trainer's saved artifacts from per-trainer folders
          3) Correctly handles BOTH:
             - Trainer.load(path) as a @classmethod returning a new instance
             - trainer.load(path) as an instance method mutating self (optionally returning self)
          4) Sets runner.model_ to something with a predict() method (prefers last trainer)
        """
        input_dir = Path(input_dir)

        # --- 1) load config ---
        cfg_path = input_dir / cls.CONFIG_FILENAME
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config: {cfg_path}")

        with open(cfg_path, "r") as f:
            cfg: Dict[str, Any] = json.load(f)

        runner = cls.from_cfg(cfg)

        # --- 2) load manifest if present (preferred) ---
        manifest_path = input_dir / cls.ARTIFACTS_FILENAME
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            entries = manifest.get("trainers", [])
        else:
            # Fallback to convention: trainer_{i}_{ClassName}
            entries = [
                {
                    "index": i,
                    "path": f"trainer_{i}_{type(tr).__name__}",
                }
                for i, tr in enumerate(runner.trainers)
            ]

        # Basic sanity check
        if len(entries) != len(runner.trainers):
            # Not always fatal, but usually indicates mismatch between saved artifacts and config
            raise ValueError(
                f"Trainer count mismatch: config created {len(runner.trainers)} trainers, "
                f"but artifact manifest has {len(entries)} entries."
            )

        # --- 3) load each trainer artifact properly ---
        for entry in entries:
            i = int(entry["index"])
            if i >= len(runner.trainers):
                raise ValueError(
                    f"Artifact manifest refers to trainer index {i}, "
                    f"but config only created {len(runner.trainers)} trainers."
                )

            trainer = runner.trainers[i]
            t_dir = input_dir / entry["path"]
            if not t_dir.exists():
                raise FileNotFoundError(f"Missing trainer artifact dir: {t_dir}")

            loaded = None

            # IMPORTANT: Prefer classmethod load (returns a new instance).
            # A @classmethod is callable on the instance too, so checking instance-first
            # can accidentally discard the loaded object.
            cls_load = getattr(type(trainer), "load", None)
            if callable(cls_load):
                loaded = cls_load(t_dir)

            else:
                inst_load = getattr(trainer, "load", None)
                if callable(inst_load):
                    loaded = inst_load(t_dir)

            # If load returns something, replace; otherwise assume in-place mutation.
            if loaded is not None:
                runner.trainers[i] = loaded

        # --- 4) set runner.model_ deterministically ---
        # Prefer the last trainer that has a callable predict() method.
        model = None
        for t in reversed(runner.trainers):
            pred = getattr(t, "predict", None)
            if callable(pred):
                model = t
                break

        final_path = input_dir / "final_model.joblib"
        if final_path.exists():
            runner.model_ = joblib.load(final_path)
            return runner

        # Fallback: last trainer
        if model is None and runner.trainers:
            model = runner.trainers[-1]

        runner.model_ = model
        return runner

    @staticmethod
    def _infer_model_from_trainers(trainers: List[Any]) -> Any:
        """
        Default policy:
          - take the last trainer that has a non-None .model_
          - else, take the last trainer itself
        """
        model = None
        for t in trainers:
            if hasattr(t, "model_") and getattr(t, "model_") is not None:
                model = getattr(t, "model_")
        if model is None and trainers:
            model = trainers[-1]
        return model

    def save_config(self, output_dir: Path, filename: str | None = None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = filename or self.CONFIG_FILENAME
        cfg = self.to_config()
        with open(output_dir / filename, "w") as f:
            json.dump(cfg, f, indent=2)
        self.log(f"Saved ModelTrainingRunner config to {output_dir / filename}")

    def save_report(self, output_dir: Path, report: Any | None = None) -> None:
        """
        Convenience wrapper around report.save(report_obj, folder).

        With PyMCRegressionReport, `report` is typically a pd.Series (metrics).
        If not provided, uses self.last_report_.
        """
        if self.report is None:
            raise RuntimeError("No report configured; cannot save_report().")

        report_obj = report if report is not None else self.last_report_
        if report_obj is None:
            raise RuntimeError("No report available to save. Run evaluate() first.")

        if not hasattr(self.report, "save") or not callable(self.report.save):
            raise AttributeError(f"{type(self.report).__name__} has no callable save()")

        self.report.save(report_obj, output_dir)
        self.log(f"Saved evaluation report to {output_dir}")
