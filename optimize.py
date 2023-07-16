from functools import partial

import wandb
import yaml
from easydict import EasyDict as edict
from fastcore.utils import Path
from loguru import logger
from skopt import gp_minimize, space

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

    param_space = [
        space.Real(0.01, 0.1, name="learning_rate"),
        space.Integer(100, 1000, name="n_estimators"),
        space.Integer(3, 15, name="max_depth"),
        space.Categorical(["gbtree", "gblinear", "dart"], name="booster"),
        # space.Categorical(["gradient_based", "uniform"], "sampling_method"),
        space.Real(1.0, 5.0, name="scale_pos_weight"),
        space.Real(0.0, 1.0, name="subsample"),
        space.Real(0.0, 1.0, name="colsample_bytree"),
        space.Real(0.0, 1.0, name="gamma"),
        space.Real(0.5, 3.0, name="reg_alpha"),
        space.Real(2.0, 5.0, name="reg_lambda"),
    ]
    param_names = [
        "learning_rate",
        "n_estimators",
        "max_depth",
        "booster",
        # "sampling_method",
        "scale_pos_weight",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ]

    optimization_function = partial(
        classifier.optimize,
        param_names=param_names,
    )

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10,
    )

    print(dict(zip(param_names, result.x)))


if __name__ == "__main__":
    optimize()
