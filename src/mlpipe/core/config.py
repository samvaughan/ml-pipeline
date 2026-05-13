from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ObjectCfg(BaseModel):
    name: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class TrainCfg(BaseModel):
    transformers: List[ObjectCfg] = Field(default_factory=list)
    trainer: ObjectCfg
    evaluators: List[ObjectCfg] = Field(default_factory=list)


class LoadDataCfg(BaseModel):
    start_date: date
    end_date: date
    sql_script_folder: Path
    source_table: Optional[str] = None
    extra_where: Optional[str] = None


class FeatureEngineeringCfg(BaseModel):
    rows_per_shard: int
    domain: Literal["racing", "sports"]


class StepOptions(BaseModel):
    load_data: Optional[LoadDataCfg] = None
    feature_engineering: Optional[FeatureEngineeringCfg] = None


class PipelineCfg(BaseModel):
    # new block for your load_data step
    step_options: StepOptions = Field(default_factory=StepOptions)

    # existing training bits
    transformers: List[ObjectCfg] = Field(default_factory=list)
    trainer: ObjectCfg
    evaluators: List[ObjectCfg] = Field(default_factory=list)
