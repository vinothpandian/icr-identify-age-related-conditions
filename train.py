from fastcore.utils import Path
from loguru import logger

import wandb
from classifiers import get_classifier_class
from config.config import load_config
from utils.wandb import init_wandb


def train():
    config = load_config()
    init_wandb(config)

    output_path = Path(config.output) / wandb.run.name
    output_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Starting training - {wandb.run.name}")
    logger.info(config)

    logger.info(f"Training {config.classifier}")
    classifier_class = get_classifier_class(config.classifier)
    classifier = classifier_class(config, output_path)
    classifier.train()


if __name__ == "__main__":
    train()
