import optuna
from xgboost import XGBClassifier
import numpy as np
from sklearn import metrics
from metrics import balanced_log_loss

from .base_classifier import BaseClassifier


class XGBoost(BaseClassifier):
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
            tree_method="gpu_hist",
            sampling_method="gradient_based",
            gpu_id=0,
            eval_metric=balanced_log_loss,
        )

        model = XGBClassifier(**params)
        
        val_losses = np.array([])

        for i, (train_idx, val_idx) in enumerate(self.kfold.split(self.X, self.stratify_df)):
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]

            X_val = self.X.iloc[val_idx]
            y_val = self.y.iloc[val_idx]

            model.fit(X_train, y_train)

            val_preds = model.predict_proba(X_val)
            val_loss = balanced_log_loss(y_val.values.ravel(), val_preds)
            
            val_losses = np.append(val_losses, val_loss)               
            

        y_test = self.y_test.values.ravel()
        preds = model.predict_proba(self.X_test)
        
        val_loss = np.mean(val_losses)
        test_loss = balanced_log_loss(y_test, preds)
        
        overfitting = metrics.mean_squared_error([val_loss], [test_loss])

        return test_loss, overfitting

    def fit(self):
        self.model = XGBClassifier(
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
