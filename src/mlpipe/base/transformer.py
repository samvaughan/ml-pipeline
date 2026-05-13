from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import joblib
import pandas as pd


class BaseTransformer(ABC):
    run_during_fit: bool = True
    requires_context_keys_for_fit: list[str] = []

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs) -> Self: ...

    @abstractmethod
    def transform(self, df: pd.DataFrame, training: bool = False, **kwargs) -> pd.DataFrame: ...

    def fit_transform(self, df: pd.DataFrame, training: bool = True, **kwargs) -> pd.DataFrame:
        return self.fit(df, **kwargs).transform(df, training=training, **kwargs)

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_dir / f"{type(self).__name__}.joblib")

    @classmethod
    def load(cls, input_dir: Path) -> Self:
        return joblib.load(Path(input_dir) / f"{cls.__name__}.joblib")

    def to_config(self) -> dict[str, Any]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement to_config(). "
            "Either implement it or ensure FeatureEngineer was built with from_cfg()."
        )
