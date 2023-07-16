from classifiers.base_classifier import BaseClassifier
from config.config import ClassifierType

from .xgboost import XGBoost


def get_classifier_class(classifier_type: ClassifierType) -> BaseClassifier:
    match classifier_type:
        case "xgboost":
            return XGBoost
        case _:
            return None
