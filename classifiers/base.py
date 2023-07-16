from abc import ABC, abstractmethod

import wandb
from sklearn import metrics

from metrics.loss import balanced_log_loss
from visualize.results import plot_results


class BaseClassifier(ABC):
    def __init__(self, clf_config, data_preprocessor, config, output_path):
        self.clf_config = clf_config
        self.data_preprocessor = data_preprocessor
        self.config = config
        self.output_path = output_path
        self.model = None
        self.data = None

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def get_preds(self, type="val"):
        pass

    def evaluate(self, pred_probs, y_true, prefix="val"):
        y_pred = pred_probs.argmax(axis=1)
        y_pred_probs = pred_probs[:, 1]

        log_loss_val = metrics.log_loss(y_true, pred_probs)
        loss_val = balanced_log_loss(y_true, pred_probs)

        accuracy_val = metrics.accuracy_score(y_true, y_pred)
        kappa_val = metrics.cohen_kappa_score(y_true, y_pred)
        f1_val = metrics.f1_score(y_true, y_pred)

        plot_results(y_true, y_pred_probs, self.output_path / "results.png")
        wandb.log({"results": wandb.Image(str(self.output_path / "results.png"))})

        wandb.log(
            {
                f"{prefix}_balanced_log_loss": loss_val,
                f"{prefix}_accuracy": accuracy_val,
                f"{prefix}_kappa": kappa_val,
                f"{prefix}_f1": f1_val,
            }
        )

        if prefix == "test":
            wandb.summary["balanced_log_loss"] = loss_val
            wandb.summary["accuracy"] = accuracy_val
            wandb.summary["kappa"] = kappa_val
            wandb.summary["f1"] = f1_val
            wandb.summary["log_loss"] = log_loss_val

    def get_validation_metrics(self):
        self.val_pred_probs, self.val_y_true = self.get_preds()
        self.evaluate(self.val_pred_probs, self.val_y_true, prefix="val")

    def get_test_data_metrics(self):
        self.test_pred_probs, self.test_y_true = self.get_preds(type="test")
        self.evaluate(self.test_pred_probs, self.test_y_true, prefix="test")

    def train(self):
        self.preprocess_data()
        self.build_model()
        self.fit()
        self.get_validation_metrics()
        self.get_test_data_metrics()
