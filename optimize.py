import optuna
from fastcore.utils import Path
from loguru import logger

import wandb
from classifiers import get_classifier_class
from config.config import load_config
from utils.wandb import init_wandb


def optimize():
    config = load_config()
    init_wandb(config)

    output_path = Path(config.output) / wandb.run.name
    output_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Optimizing {config.classifier} using Optuna")
    classifier_class = get_classifier_class(config.classifier)
    classifier = classifier_class(config, output_path)

    study = optuna.create_study(
        direction="minimize",
        study_name="catboost_symmetric_tree",
        storage="sqlite:///optuna_catboost.db",
        load_if_exists=True,
    )
    study.optimize(classifier.optimize, n_trials=500)


if __name__ == "__main__":
    optimize()
