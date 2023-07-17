from catboost import CatBoostClassifier

from metrics import balanced_log_loss

from .base_classifier import BaseClassifier


class CatBoost(BaseClassifier):
    def objective(self, trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 100, 1000, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, step=0.5),
            depth=trial.suggest_int("depth", 3, 8),
            random_strength=trial.suggest_float("random_strength", 0.0, 100.0, step=2.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0, step=0.5),
            border_count=trial.suggest_int("border_count", 1, 255),
            grow_policy=trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
            verbose=0,
            task_type="GPU",
            devices="0",
        )

        # bayesian_params = dict(
        #     bootstrap_type="bayesian",
        #     bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0, step=0.5),
        #     sampling_unit="observation",
        # )

        model = CatBoostClassifier(**params)

        for idx in self.kfold.split(self.X, self.stratify_df):
            train_idx, _ = idx
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]

            model.fit(X_train, y_train)

        y_test = self.y_test.values.ravel()
        preds = model.predict_proba(self.X_test)

        return balanced_log_loss(y_test, preds)

    def fit(self):
        self.model = CatBoostClassifier(
            **self.config.model_kwargs,
            loss_function=balanced_log_loss,
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
