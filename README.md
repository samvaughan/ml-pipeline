# ml-pipeline

A registry-based, YAML-configurable framework for building reproducible ML pipelines. It provides the orchestration skeleton; you supply the components.

## Concepts

**ml-pipeline provides:**
- Abstract base classes (`BaseTransformer`, `BaseTrainer`)
- A `Registry` for registering and instantiating components by name
- Three pipeline stages: `FeatureEngineer`, `DataFinaliser`, `ModelTrainingRunner`
- A way to construct YAML-driven pipelines

**Your project provides:**
- Custom transformers, trainers, evaluators, cleaners, and selectors
- Data loading and SQL templates
- YAML config files that wire components together by name
- Any cloud/infra-specific code (AzureML, Snowflake, etc.)

---

## Project structure

```
my-project/
├── src/myproject/
│   ├── registries.py          # defines Registry instances
│   ├── transformers/          # @TRANSFORMERS.register("name")
│   ├── trainers/              # @TRAINERS.register("name")
│   ├── evaluators/            # @EVALUATORS.register("name")
│   ├── selectors/             # @SELECTORS.register("name")
│   └── pipeline.py            # wires registries into from_cfg calls
├── configs/
│   └── pipeline.yaml
└── pyproject.toml             # ml-pipeline as a dependency
```

---

## Step 1 — Define registries

Create a single `registries.py` file in your project:

```python
from mlpipe.core.registry import Registry

CLEANERS     = Registry("cleaner")
TRANSFORMERS = Registry("transformer")
TRAINERS     = Registry("trainer")
EVALUATORS   = Registry("evaluator")
SELECTORS    = Registry("selector")
```

---

## Step 2 — Implement components

Each component extends a base class and registers itself with a decorator:

```python
from mlpipe.base.transformer import BaseTransformer
from myproject.registries import TRANSFORMERS

@TRANSFORMERS.register("rolling_features")
class RollingFeatures(BaseTransformer):
    def __init__(self, window: int):
        self.window = window

    def fit(self, df, **kwargs):
        return self

    def transform(self, df, training=False, **kwargs):
        # compute features ...
        return df
```

The same pattern applies for trainers (`BaseTrainer`), evaluators, cleaners, and selectors.

---

## Step 3 — Write a YAML config

Components are referenced by their registered name. kwargs are passed directly to `__init__`.

```yaml
feature_engineering:
  cleaners:
    - name: drop_nulls
  transformers:
    - name: rolling_features
      kwargs: {window: 4}
    - name: lag_features
      kwargs: {lags: [1, 2, 4]}

data_finaliser:
  train_test_split_column: split
  id_column: user_id          # omit if no row filtering needed
  excluded_ids: []
  output_column_spec:
    feature_columns: [rolling_4w, lag_1w, lag_2w]
    target_column: target
    metadata_columns: [user_id, week]
  selectors:
    train: {name: all_rows}
    test:  {name: all_rows}

training:
  trainers:
    - name: xgboost_regressor
      kwargs: {n_estimators: 500}
  evaluators:
    - name: rmse_report
```

---

## Step 4 — Wire it together

Pass your yaml file to the pipeline constructors:

```python
import yaml
from mlpipe.pipelines.feature_engineer import FeatureEngineer
from mlpipe.pipelines.data_finaliser import DataFinaliser
from mlpipe.pipelines.trainer import ModelTrainingRunner

# Import component modules so @register decorators run before from_cfg is called.
# See autodiscovery below for a less manual alternative.
import myproject.transformers.rolling_features
import myproject.trainers.xgboost_regressor
import myproject.evaluators.rmse_report

with open("configs/pipeline.yaml") as f:
    cfg = yaml.safe_load(f)

from myproject.registries import CLEANERS, TRANSFORMERS, TRAINERS, EVALUATORS, SELECTORS

fe = FeatureEngineer.from_cfg(cfg, cleaners=CLEANERS, transformers=TRANSFORMERS)
df_final = DataFinaliser.from_cfg(cfg, transformers=TRANSFORMERS, selectors=SELECTORS)
runner = ModelTrainingRunner.from_cfg(cfg, trainers=TRAINERS, evaluators=EVALUATORS)
```

---

## Autodiscovery

To avoid maintaining a manual import list, use `core/loader.py` to auto-import all modules under a package — triggering every `@register` decorator automatically.

```python
from mlpipe.core.loader import autodiscover

autodiscover("myproject.transformers")  # imports all modules, triggering @register decorators
autodiscover("myproject.trainers")
autodiscover("myproject.evaluators")
```

You can also pass a module object directly:

```python
import myproject.transformers
autodiscover(myproject.transformers)
```

---

## Installation

Add ml-pipeline as a dependency in your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "ml-pipeline @ git+https://github.com/samvaughan/ml-pipeline.git",
    # or a local editable install:
    # "ml-pipeline @ file:///path/to/ml-pipeline"
]
```
