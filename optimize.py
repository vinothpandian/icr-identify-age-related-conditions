import optuna
import wandb
import yaml
from easydict import EasyDict as edict
from fastcore.utils import Path
from loguru import logger

from classifiers import get_classifier_class
from preprocessor.preprocessor import Preprocessor
from utils.wandb import init_wandb


def optimize():
    with open("./config.yaml", "r") as f:
        config_dictionary = yaml.safe_load(f)
        config = edict(config_dictionary)

    init_wandb(config_dictionary)

    output_path = Path(config.output) / wandb.run.name
    output_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Starting training - {wandb.run.name}")
    logger.info(config)

    logger.info("Loading preprocess")

    data_preprocessor = Preprocessor(config)
    data_preprocessor.preprocess()

    clf_config = config.classifier

    logger.info(f"Optimizing {clf_config.name} using skopt")
    classifier_class = get_classifier_class(clf_config.type)
    classifier = classifier_class(
        clf_config,
        data_preprocessor,
        config,
        output_path,
    )
    classifier.preprocess_data()

    study = optuna.create_study(
        direction="minimize", storage="sqlite:///optuna_xgboost.db"
    )
    study.optimize(classifier.optimize, n_trials=15)


if __name__ == "__main__":
    optimize()
