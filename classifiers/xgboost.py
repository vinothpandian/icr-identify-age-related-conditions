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

    def optimize(self, params):
        model = XGBClassifier(**params, eval_metric=balanced_log_loss)

        kfold = model_selection.RepeatedStratifiedKFold(
            **self.clf_config.kfold_kwargs,
        )

        balanced_log_loss_scores = np.array([])

        for idx in kfold.split(self.X, self.y):
            train_idx, val_idx = idx
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict_proba(X_val)
            fold_score = balanced_log_loss(y_val.values.ravel(), preds)
            balanced_log_loss_scores = np.append(
                balanced_log_loss_scores,
                fold_score,
            )

        return balanced_log_loss_scores.mean()

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
