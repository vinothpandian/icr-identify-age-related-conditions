from pathlib import Path

import optuna
import pandas as pd
from fastai.tabular.core import cont_cat_split
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

from metrics.loss import balanced_log_loss
from preprocessor.utils import get_preprocess_pipeline

path = Path("./data")
output_path = Path("./submission")


def resample(X, y):
    sampler = SMOTE()
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res


train_df = pd.read_csv(path / "train.csv", index_col="Id")

drop_cols = ["EJ"]
dep_vars = ["Class"]

untrainable_cols = drop_cols + dep_vars

# Drops the dep_vars before splitting categorical and continuous variables
cont_names, cat_names = cont_cat_split(train_df, dep_var=untrainable_cols)

train_df, test_df = model_selection.train_test_split(train_df, test_size=0.4, random_state=33)
train_df.shape, test_df.shape


preprocessor = get_preprocess_pipeline(train_df, cont_names, cat_names, untrainable_cols)
X_pre = preprocessor.fit_transform(train_df.drop(columns=untrainable_cols))
train_df = pd.merge(X_pre, train_df[untrainable_cols], left_index=True, right_index=True)


X_test = preprocessor.transform(test_df.drop(columns=untrainable_cols, errors="ignore"))
y_test = test_df[dep_vars]


X = train_df.drop(columns=untrainable_cols, errors="ignore")
y = train_df[dep_vars]


xgm_clf = XGBClassifier(
    objective="binary:logistic",
    colsample_bylevel=0.3,
    colsample_bynode=0.7,
    colsample_bytree=1.0,
    gamma=0.6,
    learning_rate=0.0344,
    max_depth=3,
    min_child_weight=0.5,
    n_estimators=650,
    reg_alpha=0.0,
    reg_lambda=0.0,
    scale_pos_weight=5.5,
    subsample=0.6,
    tree_method="hist",
    eval_metric=balanced_log_loss,
)

lgbm_clf = LGBMClassifier(
    objective="binary",
    verbosity=-1,
    boosting_type="gbdt",
    learning_rate=0.046,
    n_estimators=300,
    max_depth=12,
    scale_pos_weight=10.0,
    subsample=0.9,
    reg_alpha=3.5,
    reg_lambda=2.5,
    colsample_bytree=1.0,
    colsample_bynode=0.6,
    data_sample_strategy="bagging",
    num_leaves=6329,
    max_bin=401,
    n_jobs=-1,
)


def optimize(trial):
    xgm_weight = trial.suggest_float("xgm_weight", 0.0, 1.0)
    lgm_weight = 1.0 - xgm_weight

    ensemble_clf = VotingClassifier(
        estimators=[
            ("xgm", xgm_clf),
            ("lgbm", lgbm_clf),
        ],
        weights=[xgm_weight, lgm_weight],
        voting="soft",
    )

    kfold = model_selection.RepeatedStratifiedKFold(n_splits=7, n_repeats=4)

    for idx in kfold.split(X, y):
        train_idx, _ = idx
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_res, y_res = resample(X_train, y_train.values.ravel())

        ensemble_clf.fit(X_res, y_res)

    pred_probs = ensemble_clf.predict_proba(X_test)
    y_true = y_test.values.ravel()
    return balanced_log_loss(y_true, pred_probs)


study = optuna.create_study(
    direction="minimize",
    study_name="voting_ensemble_xgm_lgbm",
    storage="sqlite:///optuna.db",
    load_if_exists=True,
)
study.optimize(optimize, n_trials=50)
