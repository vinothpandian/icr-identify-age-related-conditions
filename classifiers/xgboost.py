from xgboost import XGBClassifier

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
        self.model = XGBClassifier(
            **self.config.model_kwargs,
            eval_metric=balanced_log_loss,
        )

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
