from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import wandb
import yaml
from catboost import CatBoostClassifier
from easydict import EasyDict as edict
from fastai.callback.wandb import WandbCallback
from fastai.tabular.all import (
    Categorify,
    CategoryBlock,
    CSVLogger,
    EarlyStoppingCallback,
    FocalLossFlat,
    Path,
    TabularDataLoaders,
    TrainTestSplitter,
    accuracy,
    cont_cat_split,
    skm_to_fastai,
    slide,
    tabular_learner,
    torch,
    valley,
)
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from lightgbm.sklearn import LGBMClassifier

# Problematic imports
from loguru import logger
from scipy import stats
from sklearn import (
    compose,
    ensemble,
    impute,
    metrics,
    model_selection,
    pipeline,
    preprocessing,
    tree,
)
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier
from BaseClassifier import BaseClassifier

from metrics import balanced_log_loss

# Fix for problematic imports
# import logging
# logger = logging.getLogger(__name__)


# wandb = edict(
#     {
#         "log": print,
#         "Image": noop,
#         "define_metric": noop,
#         "init": noop,
#         "run": edict(
#             {
#                 "name": "test",
#             }
#         ),
#         "plots": edict(
#             {
#                 "precision_recall": noop,
#             }
#         ),
#         "summary": edict({}),
#         "finish": noop,
#     }
# )


def init_wandb(config_dictionary):
    wandb.init(
        project="icr-identify-age-related-conditions",
        config=config_dictionary,
        mode=config_dictionary.get("mode", "offline"),
    )
    wandb.define_metric("val_balanced_log_loss", summary="min")
    wandb.define_metric("val_accuracy", summary="max")
    wandb.define_metric("val_kappa", summary="max")
    wandb.define_metric("val_f1", summary="max")

    wandb.define_metric("test_balanced_log_loss", summary="min")
    wandb.define_metric("test_accuracy", summary="max")
    wandb.define_metric("test_kappa", summary="max")
    wandb.define_metric("test_f1", summary="max")


def get_selected_features(type, df, dep_vars, model_kwargs):
    model = None

    match type:
        case "decision_tree":
            model = tree.DecisionTreeClassifier(**model_kwargs)
        case "random_forest":
            model = ensemble.RandomForestClassifier(**model_kwargs)
        case "gboost":
            model = ensemble.GradientBoostingClassifier(**model_kwargs)
        case _:
            model = tree.ExtraTreeClassifier(**model_kwargs)

    X = df.drop(columns=dep_vars)
    y = df[dep_vars]
    model.fit(X, y)
    imp_df = pd.DataFrame(dict(cols=X.columns, imp=model.feature_importances_))
    imp_df = imp_df[imp_df.imp > 0.005]
    return imp_df.cols.values


def get_sampling_strategy(sampling_strategy):
    match sampling_strategy:
        case "random_over_sampler":
            return RandomOverSampler
        case "smote":
            return SMOTE
        case "svmsmote":
            return SVMSMOTE
        case "adasyn":
            return ADASYN
        case "kmeans_smote":
            return KMeansSMOTE
        case _:
            return None


def get_preprocess_pipeline(df, cont_names, dep_vars):
    """
    Returns a pipeline that performs the following transformations:
    * Standard scaling
    * Log transformation
    * Reciprocal transformation
    * Box-Cox transformation
    * Yeo-Johnson transformation
    * Categorical imputing
    * Semi-constant feature binarization

    Based on the EDA from https://www.kaggle.com/code/mateuszk013/icr-eda-balanced-learning-with-lgbm-xgb/notebook

    :param df: The dataframe to be transformed.
    :type df: pandas.DataFrame
    :param cont_names: The names of the continuous variables.
    :type cont_names: list of str
    :param dep_vars: The names of the dependent variables.
    :type dep_vars: list of str
    """

    # Identify columns that doesn't follow a normal distribution
    # find an appropriate transformation for them to follow a normal distribution
    r2_scores = defaultdict(tuple)

    for feature in cont_names:
        orig = df[feature].dropna()
        _, (*_, R_orig) = stats.probplot(orig, rvalue=True)
        _, (*_, R_log) = stats.probplot(np.log(orig), rvalue=True)
        _, (*_, R_sqrt) = stats.probplot(np.sqrt(orig), rvalue=True)
        _, (*_, R_reci) = stats.probplot(np.reciprocal(orig), rvalue=True)
        _, (*_, R_boxcox) = stats.probplot(stats.boxcox(orig)[0], rvalue=True)
        _, (*_, R_yeojohn) = stats.probplot(stats.yeojohnson(orig)[0], rvalue=True)
        r2_scores[feature] = (
            R_orig * R_orig,
            R_log * R_log,
            R_sqrt * R_sqrt,
            R_reci * R_reci,
            R_boxcox * R_boxcox,
            R_yeojohn * R_yeojohn,
        )

    r2_scores = pd.DataFrame(
        r2_scores,
        index=("Original", "Log", "Sqrt", "Reciprocal", "BoxCox", "YeoJohnson"),
    ).T

    r2_scores["Winner"] = r2_scores.idxmax(axis=1)

    # Identify columns to be transformed
    no_transform_cols = r2_scores.query("Winner == 'Original'").index
    log_transform_cols = r2_scores.query("Winner == 'Log'").index
    reciprocal_transform_cols = r2_scores.query("Winner == 'Reciprocal'").index
    boxcox_transform_cols = r2_scores.query("Winner == 'BoxCox'").index
    yeojohnson_transform_cols = r2_scores.query("Winner == 'YeoJohnson'").index

    # Identify columns that are constant or semi-constant
    numeric_descr = df.drop(columns=dep_vars).describe().T
    semi_constant_mask = np.isclose(numeric_descr["min"], numeric_descr["50%"])
    semi_constant_descr = numeric_descr[semi_constant_mask]
    semi_const_cols_thresholds = semi_constant_descr["50%"].to_dict()

    # List of columns to be transformed
    semi_const_cols = semi_const_cols_thresholds.keys()
    no_transform_cols = no_transform_cols.drop(
        semi_const_cols, errors="ignore"
    ).to_list()
    log_transform_cols = log_transform_cols.drop(
        semi_const_cols, errors="ignore"
    ).to_list()
    reciprocal_transform_cols = reciprocal_transform_cols.drop(
        semi_const_cols, errors="ignore"
    ).to_list()
    boxcox_transform_cols = boxcox_transform_cols.drop(
        semi_const_cols, errors="ignore"
    ).to_list()
    yeojohnson_transform_cols = yeojohnson_transform_cols.drop(
        semi_const_cols, errors="ignore"
    ).to_list()

    # Transformations
    standard_scaling = (
        preprocessing.StandardScaler(),
        no_transform_cols,
    )
    log_transform = (
        pipeline.make_pipeline(
            preprocessing.FunctionTransformer(
                func=np.log, feature_names_out="one-to-one"
            ),
            preprocessing.StandardScaler(),
        ),
        log_transform_cols,
    )
    reciprocal_transform = (
        pipeline.make_pipeline(
            preprocessing.FunctionTransformer(
                func=np.reciprocal, feature_names_out="one-to-one"
            ),
            preprocessing.StandardScaler(),
        ),
        reciprocal_transform_cols,
    )
    boxcox_transform = (
        preprocessing.PowerTransformer(method="box-cox", standardize=True),
        boxcox_transform_cols,
    )
    yeojohnson_transform = (
        preprocessing.PowerTransformer(method="yeo-johnson", standardize=True),
        yeojohnson_transform_cols,
    )

    # Other transformations
    categorical_imputing = (
        pipeline.make_pipeline(
            impute.SimpleImputer(strategy="most_frequent"),
            preprocessing.OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ),
        ),
        compose.make_column_selector(dtype_include=object),  # type: ignore
    )
    semi_const_transforms = [
        (
            pipeline.make_pipeline(
                impute.SimpleImputer(strategy="median"),
                preprocessing.Binarizer(threshold=thresh),
            ),
            [col],
        )
        for col, thresh in semi_const_cols_thresholds.items()
    ]

    return pipeline.make_pipeline(
        compose.make_column_transformer(
            standard_scaling,
            log_transform,
            reciprocal_transform,
            boxcox_transform,
            yeojohnson_transform,
            categorical_imputing,
            *semi_const_transforms,
            remainder="drop",
            verbose_feature_names_out=False,
        ),
        impute.KNNImputer(n_neighbors=10, weights="distance"),
    ).set_output(transform="pandas")


def plot_results(y_true, y_pred, output_file_path):
    """
    Plots the predictions of a model against the true values.
    It's useful to visualize the predictions of a model in a binary classification problem.


    :param y_true: The true values of the target variable.
    :type y_true: Numpy array of size n_samples
    :param y_pred: _description_
    :type y_pred: Numpy array of size n_samples
    :param output_file_path: The path to save the plot.
    :type output_file_path: Path
    """
    y_proba_frame = pd.DataFrame(
        {
            "Sample Integer Index": np.arange(0, len(y_true)),
            "Positive Class Probability": y_pred,
            "Class": y_true.astype(str),
        },
    )

    fig = px.scatter(
        y_proba_frame,
        x="Positive Class Probability",
        y="Sample Integer Index",
        symbol="Class",
        symbol_sequence=["diamond", "circle"],
        color="Class",
        color_discrete_sequence=["#010D36", "#FF2079"],
        category_orders={"Class": ("0", "1")},
        opacity=0.6,
        height=540,
        width=840,
        title="Training Dataset - Out of Fold Predictions",
    )
    fig.update_layout(
        title_font_size=18,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="right",
            y=1.05,
            x=1,
            title="Class",
            itemsizing="constant",
        ),
        xaxis_range=[-0.02, 1.02],
    )
    fig.update_traces(marker_size=6)
    fig.write_image(str(output_file_path))


def balanced_log_loss(y_pred, y_true):
    """
    Computes the balanced log loss for a binary classification problem.
    Competition metric for the ICR Identify Age-Related Conditions competition.

    :param y_pred: Predictions of the model. The first column is the probability
    of the negative class and the second column is the probability of the positive class.
    :type y_pred: Torch tensor of size n_samples x 2
    :param y_true: The true values of the target variable.
    :type y_true: Torch tensor of size n_samples
    :return: The balanced log loss.
    :rtype: float
    """

    y1 = y_true.flatten()
    y0 = 1 - y1

    n0, n1 = torch.bincount(y1, minlength=2)

    p0 = torch.clip(y_pred[:, 0], 1e-15, 1 - 1e-15)
    p1 = torch.clip(y_pred[:, 1], 1e-15, 1 - 1e-15)

    log_loss_0 = (y0 * torch.log(p0)).sum()
    log_loss_1 = (y1 * torch.log(p1)).sum()

    loss_0 = (
        -1 / n0 * log_loss_0
        if log_loss_0 != 0
        else torch.tensor(0.0, requires_grad=True)
    )
    loss_1 = (
        -1 / n1 * log_loss_1
        if log_loss_1 != 0
        else torch.tensor(0.0, requires_grad=True)
    )

    loss_score = (loss_0 + loss_1) / 2

    return loss_score


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.path = Path(self.config.root)
        self.train_df = None
        self.test_df = None
        self.cont_names = None
        self.cat_names = None
        self.dep_vars = None

    def load_data(self):
        self.drop_vars = self.config.train.drop_vars
        self.train_df = pd.read_csv(self.path / self.config.train.file, index_col="Id")
        self.train_df.drop(columns=self.drop_vars, inplace=True)

        self.dep_vars = self.config.train.dep_vars
        self.cont_names, self.cat_names = cont_cat_split(
            self.train_df, dep_var=self.dep_vars
        )
        self.test_df = pd.read_csv(self.path / self.config.test.file, index_col="Id")
        self.test_df.drop(columns=self.drop_vars, inplace=True)

        # Dummy protection for an empty test dataset
        if np.all(np.isclose(self.test_df.select_dtypes("number").sum(), 0)):
            self.test_df[self.cont_names] += 1e-9

    def preliminary_preprocessing(self):
        preprocessor = get_preprocess_pipeline(
            self.train_df, self.cont_names, self.dep_vars
        )
        self.X_pre = preprocessor.fit_transform(
            self.train_df.drop(columns=self.dep_vars)
        )
        self.train_df = pd.merge(
            self.X_pre, self.train_df[self.dep_vars], left_index=True, right_index=True
        )
        test_data = preprocessor.transform(self.test_df.drop(columns=self.dep_vars))
        self.test_df = pd.merge(
            test_data, self.test_df[self.dep_vars], left_index=True, right_index=True
        )

    def resample(self):
        if sampler := get_sampling_strategy(self.config.train.get("sampling_strategy")):
            logger.info(f"Resampling data with {self.config.train.sampling_strategy}")
            sampler_kwargs = self.config.train.get("sampling_strategy_kwargs", {})
            X_res, y_res = sampler(**sampler_kwargs).fit_resample(
                self.X_pre, self.train_df[self.dep_vars]
            )
            self.train_df = pd.merge(X_res, y_res, left_index=True, right_index=True)

    def select_features(self):
        if feature_selector := self.config.train.get("feature_selector"):
            selector_kwargs = self.config.train.get("feature_selector_kwargs", {})
            features = get_selected_features(
                feature_selector, self.train_df, self.dep_vars, selector_kwargs
            )
            selected_features = [*features, *self.dep_vars]
            self.train_df = self.train_df[selected_features]
            self.test_df = self.test_df[selected_features]

            self.cont_names = [
                col for col in self.cont_names if col in selected_features
            ]
            self.cat_names = [col for col in self.cat_names if col in selected_features]

    def preprocess(self):
        self.load_data()
        self.preliminary_preprocessing()
        self.resample()
        self.select_features()

        return self.train_df, self.test_df, self.cont_names, self.cat_names


class TabularLearner(BaseClassifier):
    def __init__(self, clf_config, data_preprocessor, config, output_path) -> None:
        super().__init__(clf_config, data_preprocessor, config, output_path)

    def preprocess_data(self):
        df = self.data_preprocessor.train_df
        cat_names = self.data_preprocessor.cat_names
        cont_names = self.data_preprocessor.cont_names

        stratify = df["Class"] if "Class" in df.columns else None

        dls = TabularDataLoaders.from_df(
            df,
            cat_names=cat_names,
            cont_names=cont_names,
            procs=[Categorify],
            y_names=self.config.train.dep_vars,
            y_block=CategoryBlock,
            splits=TrainTestSplitter(
                test_size=self.config.train.val_split,
                stratify=stratify,
            ),
            bs=self.config.batch_size,
        )
        self.data = dls
        return self.data

    def build_model(self):
        loss_func = FocalLossFlat(**self.clf_config.loss_kwargs)

        early_stopping = EarlyStoppingCallback(
            monitor="valid_loss",
            comp=np.less,
            min_delta=self.clf_config.early_stopping_min_delta,
            patience=self.clf_config.early_stopping_patience,
        )
        csv_logger = CSVLogger(fname="history.csv")

        cbs = [early_stopping, csv_logger]

        if wandb.run.name != "test":
            wandb_logger = WandbCallback(log="all", log_preds=False)
            cbs += [wandb_logger]

        self.model = tabular_learner(
            self.data,
            metrics=[accuracy, self.kappa, self.f1, skm_to_fastai(metrics.log_loss)],
            loss_func=loss_func,
            cbs=cbs,
            layers=self.clf_config.layers,
            path=self.output_path,
        )
        return self.model

    def fit(self):
        if self.clf_config.lr_find:
            values = self.model.lr_find(suggest_funcs=(slide, valley))
            wandb.log({"lr_plot": plt})
            for key, value in values._asdict().items():
                wandb.log({key: value})

        fit_func = getattr(self.clf_config, "fit_func", None)

        if fit_func and callable(func := getattr(self.model, fit_func)):
            func(**self.clf_config.fit_kwargs)
        else:
            self.model.fit_one_cycle(**self.clf_config.fit_kwargs)

    def get_preds(self, type="val"):
        if type == "val":
            pred_probs, y_true = self.model.get_preds()
            return pred_probs.numpy(), y_true.ravel().numpy()

        test_df = self.data_preprocessor.test_df
        test_dl = self.model.dls.test_dl(test_df, with_labels=True)
        pred_probs, y_true = self.model.get_preds(dl=test_dl)
        return pred_probs.numpy(), y_true.ravel().numpy()


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
        df.to_csv("grid_search_results.csv", index=False)
        df["overfitting_metric"] = (
            df["mean_test_val_balanced_log_loss"]
            - df["mean_test_test_balanced_log_loss"]
        )
        return df.overfitting_metric.idxmax()

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
            n_jobs=32,
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
        df["overfitting_metric"] = (
            df["mean_test_val_balanced_log_loss"]
            - df["mean_test_test_balanced_log_loss"]
        )
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


class LightGBoost(BaseClassifier):
    def preprocess_data(self):
        df = self.data_preprocessor.train_df
        self.X = df.drop(columns=self.config.train.dep_vars)
        self.y = df[self.config.train.dep_vars]

        test_df = self.data_preprocessor.test_df
        self.X_test = test_df.drop(columns=self.config.train.dep_vars)
        self.y_test = test_df[self.config.train.dep_vars]
        self.scorer = metrics.make_scorer(
            balanced_log_loss, greater_is_better=False, needs_proba=True
        )

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


def get_classifier_class(classifier_type):
    match classifier_type:
        case "tabular_learner":
            return TabularLearner
        case "xgboost":
            return XGBoost
        case "lightgbm":
            return LightGBoost
        case "catboost":
            return CatBoost
        case "tabpfn":
            return TabP
        case _:
            return None


def train():
    with open("./config.yaml", "r") as f:
        config_dictionary = yaml.safe_load(f)
        config = edict(config_dictionary)

    init_wandb(config_dictionary)

    output_path = Path(config.output) / wandb.run.name
    output_path.mkdir(exist_ok=True)

    logger.info(f"Starting training - {wandb.run.name}")
    logger.info(config)

    logger.info("Loading data")

    data_preprocessor = DataPreprocessor(config)
    data_preprocessor.preprocess()

    clf_config = config.classifier

    logger.info(f"Training {clf_config.name}")
    classifier_class = get_classifier_class(clf_config.type)
    classifier = classifier_class(clf_config, data_preprocessor, config, output_path)
    classifier.train()


if __name__ == "__main__":
    train()
