from classifiers.base_classifier import BaseClassifier
from config.config import ClassifierType

from .lightgbm import LightGBM
from .xgboost import XGBoost


def get_classifier_class(classifier_type: ClassifierType) -> BaseClassifier:
    match classifier_type:
        case "xgboost":
            return XGBoost
        case "lightgbm":
            return LightGBM
        case _:
            return None
