from classifiers.base_classifier import BaseClassifier
from classifiers.catboost import CatBoost
from config.config import ClassifierType

from .lightgbm import LightGBM
from .xgboost import XGBoost


def get_classifier_class(classifier_type: ClassifierType) -> BaseClassifier:
    if classifier_type == "xgboost":
        return XGBoost
    elif classifier_type == "lightgbm":
        return LightGBM
    elif classifier_type == "catboost":
        return CatBoost
    else:
        return None
