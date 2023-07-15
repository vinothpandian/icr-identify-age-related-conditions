from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from .base import BaseClassifier
from metrics import balanced_log_loss


class TabP(BaseClassifier):
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
        self.model = TabPFNClassifier(**self.clf_config.model_kwargs)

    def fit(self):
        self.model.fit(self.X, self.y)
        self.val_pred_probs = self.model.predict_proba(self.X)

    def get_preds(self, type="val"):
        if type == "val":
            return self.val_pred_probs, self.y.values.ravel()

        pred_probs = self.model.predict_proba(self.X_test)
        return pred_probs, self.y_test.values.ravel()
