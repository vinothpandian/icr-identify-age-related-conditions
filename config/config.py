from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import yaml

ClassifierType = Literal["xgboost", "lightgbm", "catboost"]
SamplingStrategyType = Literal["random_over_sampler", "smote", "svmsmote", "adasyn", "kmeans_smote"]


@dataclass
class Config:
    root: str
    output: str
    mode: str = "offline"
    is_submission: bool = False
    test: Dict[str, Any] = field(default_factory=dict)
    dep_vars: List[str] = field(default_factory=list)
    with_greeks: bool = False
    kfold_type: str = "stratified"
    stratify_by: List[str] = None
    sampling_strategy: SamplingStrategyType = None
    sampling_strategy_kwargs: Dict[str, Any] = field(default_factory=dict)
    classifier: ClassifierType = "xgboost"
    drop_cols: List[str] = field(default_factory=list)
    kfold_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


def load_config(file_path="config.yaml") -> Config:
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)
