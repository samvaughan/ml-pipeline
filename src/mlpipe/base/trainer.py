from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import joblib
import pandas as pd


class BaseTrainer(ABC):

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Any, *, base_artifact: Any = None, **kwargs) -> Any: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame: ...

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_dir / f"{type(self).__name__}.joblib")

    @classmethod
    def load(cls, input_dir: Path) -> Self:
        return joblib.load(Path(input_dir) / f"{cls.__name__}.joblib")

    def to_config(self) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not implement to_config().")
