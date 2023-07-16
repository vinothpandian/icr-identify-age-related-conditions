import numpy as np
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

        self.kfold = model_selection.RepeatedStratifiedKFold(**self.clf_config.kfold_kwargs)

    def build_model(self):
        self.model = XGBClassifier(
            **self.clf_config.model_kwargs,
            eval_metric=balanced_log_loss,
        )

    def objective(self, trial):
        params = dict(
            booster="gbtree",
            learning_rate=trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 15, step=1),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 10.0, step=0.5),
            subsample=trial.suggest_float("subsample", 0.0, 1.0, step=0.1),
            gamma=trial.suggest_float("gamma", 0.0, 1.0, step=0.1),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0, step=0.5),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0, step=0.5),
            min_child_weight=trial.suggest_float("min_child_weight", 0.0, 20.0, step=0.5),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.0, 1.0, step=0.1),
            colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.0, 1.0, step=0.1),
            colsample_bynode=trial.suggest_float("colsample_bynode", 0.0, 1.0, step=0.1),
            tree_method=trial.suggest_categorical("tree_method", ["auto", "exact", "approx", "hist"]),
        )

        model = XGBClassifier(
            **params,
            eval_metric=balanced_log_loss,
        )

        for idx in self.kfold.split(self.X, self.y):
            train_idx, _ = idx
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]

            model.fit(X_train, y_train)

        y_test = self.y_test.values.ravel()
        preds = model.predict_proba(self.X_test)

        return balanced_log_loss(y_test, preds)

    def fit(self):
        for idx in self.kfold.split(self.X, self.y):
            train_idx, _ = idx
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]

            self.model.fit(X_train, y_train)

        self.val_pred_probs = self.model.predict_proba(self.X)

    def get_preds(self, type="val"):
        if type == "val":
            return self.val_pred_probs, self.y.values.ravel()

        pred_probs = self.model.predict_proba(self.X_test)
        return pred_probs, self.y_test.values.ravel()
