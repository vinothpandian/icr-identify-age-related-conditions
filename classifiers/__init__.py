from .cat_boost import CatBoost
from .fastai_tabular import FastAITabularClassifier
from .lightg_boost import LightGBoost
from .tabpfn import TabP
from .xgboost import XGBoost


def get_classifier_class(classifier_type):
    match classifier_type:
        case "tabular_learner":
            return FastAITabularClassifier
        case "xgboost":
            return XGBoost
        case "lightgbm":
            return LightGBoost
        case "catboost":
            return CatBoost
        case "tabpfn":
            return TabP
        case _:
            return None
