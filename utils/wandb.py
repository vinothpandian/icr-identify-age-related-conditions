import wandb


def init_wandb(config_dictionary):
    wandb.init(
        project="icr-identify-age-related-conditions",
        config=config_dictionary,
        mode=config_dictionary.get("mode", "offline"),
    )
    wandb.define_metric("val_balanced_log_loss", summary="min")
    wandb.define_metric("val_accuracy", summary="max")
    wandb.define_metric("val_kappa", summary="max")
    wandb.define_metric("val_f1", summary="max")

    wandb.define_metric("test_balanced_log_loss", summary="min")
    wandb.define_metric("test_accuracy", summary="max")
    wandb.define_metric("test_kappa", summary="max")
    wandb.define_metric("test_f1", summary="max")
