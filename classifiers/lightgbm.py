import warnings

from lightgbm import LGBMClassifier
from sklearn.exceptions import DataConversionWarning

from metrics import balanced_log_loss

from .base_classifier import BaseClassifier

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


class LightGBM(BaseClassifier):
    def objective(self, trial):
        params = dict(
            boosting_type="gbdt",
            learning_rate=trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 15, step=1),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 10.0, step=0.5),
            subsample=trial.suggest_float("subsample", 0.1, 1.0, step=0.1),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0, step=0.5),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0, step=0.5),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0, step=0.1),
            colsample_bynode=trial.suggest_float("colsample_bynode", 0.1, 1.0, step=0.1),
            data_sample_strategy="bagging",
            num_leaves=trial.suggest_int("num_leaves", 1, 131072),
            max_bin=trial.suggest_int("max_bin", 1, 1000),
            n_jobs=-1,
        )

        model = LGBMClassifier(**params, objective="binary", verbosity=-1)

        for idx in self.kfold.split(self.X, self.stratify_df):
            train_idx, _ = idx
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]

            model.fit(
                X_train,
                y_train,
                eval_metric=balanced_log_loss,
            )

        y_test = self.y_test.values.ravel()
        preds = model.predict_proba(self.X_test)

        return balanced_log_loss(y_test, preds)

    def fit(self):
        self.model = LGBMClassifier(
            **self.config.model_kwargs,
            eval_metric=balanced_log_loss,
        )

        for idx in self.kfold.split(self.X, self.stratify_df):
            train_idx, _ = idx
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]

            X_res, y_res = self.data_preprocessor.resample(X_train, y_train)

            self.model.fit(X_res, y_res)

        self.train_pred_probs = self.model.predict_proba(self.X)

    def get_preds(self, type="val"):
        if type == "val":
            return self.train_pred_probs, self.y.values.ravel()

        if self.is_submission:
            return self.model.predict_proba(self.X_test)

        pred_probs = self.model.predict_proba(self.X_test)
        return pred_probs, self.y_test.values.ravel()
