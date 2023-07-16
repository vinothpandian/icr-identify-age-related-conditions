import pandas as pd
import wandb
from lightgbm import LGBMClassifier
from sklearn import metrics, model_selection

from .base import BaseClassifier
from metrics import balanced_log_loss


class LightGBoost(BaseClassifier):
    def preprocess_data(self):
        df = self.data_preprocessor.train_df
        self.X = df.drop(columns=self.config.train.dep_vars)
        self.y = df[self.config.train.dep_vars]

        test_df = self.data_preprocessor.test_df
        self.X_test = test_df.drop(columns=self.config.train.dep_vars)
        self.y_test = test_df[self.config.train.dep_vars]
        self.scorer = metrics.make_scorer(balanced_log_loss, greater_is_better=False, needs_proba=True)

    def build_model(self):
        self.model = LGBMClassifier(**self.clf_config.model_kwargs)

    def fit(self):
        folds = 5

        self.kfold = model_selection.StratifiedKFold(n_splits=folds)

        y = self.y.values.ravel()

        random_search = model_selection.GridSearchCV(
            self.model,
            param_grid=self.clf_config.grid_search,
            # param_distributions=self.clf_config.grid_search,
            n_jobs=32,
            scoring=self.scorer,
            cv=self.kfold.split(self.X, y),
            verbose=3,
        )

        random_search.fit(self.X, y)

        wandb.log({"Best Score": random_search.best_score_})
        wandb.summary["Best hyperparameters"] = random_search.best_params_

        results = pd.DataFrame(random_search.cv_results_)
        wandb.log({"Grid Search Results": wandb.Table(dataframe=results)})

        self.model = random_search.best_estimator_
        self.val_pred_probs = self.model.predict_proba(self.X)

    def get_preds(self, type="val"):
        if type == "val":
            return self.val_pred_probs, self.y.values.ravel()

        pred_probs = self.model.predict_proba(self.X_test)
        return pred_probs, self.y_test.values.ravel()
