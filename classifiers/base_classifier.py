from abc import ABC, abstractmethod
from pathlib import Path

from sklearn import metrics, model_selection

import wandb
from config.config import Config
from metrics.loss import balanced_log_loss
from preprocessor.preprocessor import Preprocessor
from visualize.results import plot_results


class BaseClassifier(ABC):
    def __init__(self, config: Config, output_path: Path) -> None:
        self.config = config
        self.output_path = output_path
        self.model = None

    def preprocess_data(self):
        self.data_preprocessor = Preprocessor(self.config)
        self.data_preprocessor.preprocess()

        self.train_df = self.data_preprocessor.train_df
        self.test_df = self.data_preprocessor.test_df
        self.is_submission = self.data_preprocessor.is_submission
        self.untrainable_cols = self.data_preprocessor.untrainable_cols
        self.drop_cols = self.data_preprocessor.drop_cols
        self.dep_vars = self.data_preprocessor.dep_vars

        stratify_by = self.data_preprocessor.stratify_by or self.dep_vars
        self.stratify_df = self.train_df[stratify_by]

        self.X = self.train_df.drop(columns=self.untrainable_cols, errors="ignore")
        self.y = self.train_df[self.dep_vars]

        if not self.is_submission:
            self.X_test = self.test_df.drop(columns=self.untrainable_cols, errors="ignore")
            self.y_test = self.test_df[self.dep_vars]

        kfold_type = self.config.kfold_type

        if kfold_type == "stratified":
            self.kfold = model_selection.RepeatedStratifiedKFold(**self.config.kfold_kwargs)
        else:
            self.kfold = model_selection.KFold(n_splits=len(self.X))

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def objective(self, trial):
        pass

    @abstractmethod
    def get_preds(self, type="val"):
        pass

    def evaluate(self, pred_probs, y_true, prefix="train"):
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

        if prefix == "train":
            wandb.summary["train_balanced_log_loss"] = loss_val

        if prefix == "test":
            wandb.summary["balanced_log_loss"] = loss_val
            wandb.summary["accuracy"] = accuracy_val
            wandb.summary["kappa"] = kappa_val
            wandb.summary["f1"] = f1_val
            wandb.summary["log_loss"] = log_loss_val

    def get_validation_metrics(self):
        self.train_pred_probs, self.y_train = self.get_preds()
        self.evaluate(self.train_pred_probs, self.y_train, prefix="train")

    def get_test_data_metrics(self):
        self.test_pred_probs, self.y_test = self.get_preds(type="test")
        self.evaluate(self.test_pred_probs, self.y_test, prefix="test")

    def train(self):
        self.preprocess_data()
        self.fit()
        self.get_validation_metrics()
        self.get_test_data_metrics()

    def optimize(self, trial):
        self.preprocess_data()
        try:
            loss = self.objective(trial)
            return loss
        except Exception:
            return None
