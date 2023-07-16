import dataclasses

import wandb
from config.config import Config


def init_wandb(config: Config):
    wandb.init(
        project="icr-identify-age-related-conditions",
        config=dataclasses.asdict(config),
        mode=config.mode,
    )

    metrics_summaries = {
        "balanced_log_loss": "min",
        "accuracy": "max",
        "kappa": "max",
        "f1": "max",
    }

    for prefix in ["val_", "test_"]:
        for metric, summary in metrics_summaries.items():
            wandb.define_metric(prefix + metric, summary=summary)
