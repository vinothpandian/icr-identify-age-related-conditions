import pandas as pd
import wandb
from catboost import CatBoostClassifier
from sklearn import model_selection

from .base import BaseClassifier
from metrics import balanced_log_loss


class CatBoost(BaseClassifier):
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
        self.model = CatBoostClassifier(**self.clf_config.model_kwargs)

    def _refit(self, csv_results):
        df = pd.DataFrame(csv_results)
        df.to_csv("grid_search_results.csv", index=False)
        df["overfitting_metric"] = df["mean_test_val_balanced_log_loss"] - df["mean_test_test_balanced_log_loss"]
        return df.overfitting_metric.idxmax()

    def fit(self):
        folds = 5

        self.kfold = model_selection.StratifiedKFold(n_splits=folds)
        y = self.y.values.ravel()

        random_search = model_selection.RandomizedSearchCV(
            self.model,
            # param_grid=self.clf_config.grid_search,
            param_distributions=self.clf_config.grid_search,
            n_iter=20,
            n_jobs=32,
            scoring=self.scorer,
            cv=self.kfold.split(self.X, y),
            verbose=1,
            refit=self._refit,
        )

        random_search.fit(self.X, y)

        result_df = pd.DataFrame(random_search.cv_results_)
        result_df["overfitting_score"] = result_df["mean_test_val_balanced_log_loss"] - result_df["mean_test_test_balanced_log_loss"]
        best_score_row = result_df.iloc[random_search.best_index_]
        print(f"{best_score_row.mean_test_val_balanced_log_loss}, {best_score_row.mean_test_test_balanced_log_loss}")

        wandb.summary["Best hyperparameters"] = random_search.best_params_
        wandb.log({"Grid Search Results": wandb.Table(dataframe=result_df)})

        self.model = random_search.best_estimator_
        self.val_pred_probs = self.model.predict_proba(self.X)

    def get_preds(self, type="val"):
        if type == "val":
            return self.val_pred_probs, self.y.values.ravel()

        pred_probs = self.model.predict_proba(self.X_test)
        return pred_probs, self.y_test.values.ravel()
