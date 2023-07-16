import numpy as np
import pandas as pd
import wandb
from sklearn import model_selection
from xgboost import XGBClassifier

from metrics import balanced_log_loss

from .base import BaseClassifier

np.int = np.int64


class XGBoost(BaseClassifier):
    def preprocess_data(self):
        df = self.data_preprocessor.train_df
        self.X = df.drop(columns=self.config.train.dep_vars)
        self.y = df[self.config.train.dep_vars]

        test_df = self.data_preprocessor.test_df
        self.X_test = test_df.drop(columns=self.config.train.dep_vars)
        self.y_test = test_df[self.config.train.dep_vars]
        self.scorer = lambda estimator, X, y: {
            "val_balanced_log_loss": balanced_log_loss(
                y,
                estimator.predict_proba(X),
            ),
            "test_balanced_log_loss": balanced_log_loss(
                self.y_test.values.ravel(),
                estimator.predict_proba(self.X_test),
            ),
        }

    def build_model(self):
        self.model = XGBClassifier(
            **self.clf_config.model_kwargs, eval_metric=balanced_log_loss
        )

    def _refit(self, csv_results):
        df = pd.DataFrame(csv_results)
        df.to_csv(self.output_path / "grid_search_results.csv", index=False)
        df["overfitting_metric"] = (
            df["mean_test_val_balanced_log_loss"]
            - df["mean_test_test_balanced_log_loss"]
        )
        return df.overfitting_metric.idxmax()

    def optimize(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 5.0)
        subsample = trial.suggest_float("subsample", 0.0, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.0, 1.0)
        gamma = trial.suggest_float("gamma", 0.0, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0.5, 3.0)
        reg_lambda = trial.suggest_float("reg_lambda", 2.0, 5.0)

        model = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            eval_metric=balanced_log_loss,
            gpu_id=0,
        )

        kfold = model_selection.RepeatedStratifiedKFold(
            **self.clf_config.kfold_kwargs,
        )

        for idx in kfold.split(self.X, self.y):
            train_idx, _ = idx
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]

            model.fit(X_train, y_train)

        y_test = self.y_test.values.ravel()
        preds = model.predict_proba(self.X_test)

        return balanced_log_loss(y_test, preds)

    def fit(self):
        self.kfold = model_selection.RepeatedStratifiedKFold(
            **self.clf_config.kfold_kwargs
        )
        y = self.y.values.ravel()

        random_search = model_selection.GridSearchCV(
            self.model,
            param_grid=self.clf_config.grid_search,
            # param_distributions=self.clf_config.grid_search,
            # n_iter=5,
            scoring=self.scorer,
            cv=self.kfold.split(self.X, y),
            verbose=3,
            refit=self._refit,
        )

        random_search.fit(self.X, y)

        result_df = pd.DataFrame(random_search.cv_results_)
        result_df["overfitting_score"] = (
            result_df["mean_test_val_balanced_log_loss"]
            - result_df["mean_test_test_balanced_log_loss"]
        )
        best_score_row = result_df.iloc[random_search.best_index_]
        print(
            f"{best_score_row.mean_test_val_balanced_log_loss}, {best_score_row.mean_test_test_balanced_log_loss}"
        )

        wandb.summary["Best hyperparameters"] = random_search.best_params_
        wandb.log({"Grid Search Results": wandb.Table(dataframe=result_df)})

        self.model = random_search.best_estimator_
        self.val_pred_probs = self.model.predict_proba(self.X)

    def get_preds(self, type="val"):
        if type == "val":
            return self.val_pred_probs, self.y.values.ravel()

        pred_probs = self.model.predict_proba(self.X_test)
        return pred_probs, self.y_test.values.ravel()
