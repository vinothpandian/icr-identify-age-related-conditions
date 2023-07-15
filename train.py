import wandb
import yaml
from easydict import EasyDict as edict
from fastcore.utils import Path
from loguru import logger

from classifiers import get_classifier_class
from preprocessor.preprocessor import Preprocessor

from utils.wandb import init_wandb


def train():
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

    logger.info(f"Training {clf_config.name}")
    classifier_class = get_classifier_class(clf_config.type)
    classifier = classifier_class(clf_config, data_preprocessor, config, output_path)
    classifier.train()


if __name__ == "__main__":
    train()
