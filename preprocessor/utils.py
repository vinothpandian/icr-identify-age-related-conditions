from collections import defaultdict

import numpy as np
import pandas as pd
from imblearn.over_sampling import (ADASYN, SMOTE, SVMSMOTE, KMeansSMOTE,
                                    RandomOverSampler)
from scipy import stats
from sklearn import compose, ensemble, impute, pipeline, preprocessing, tree


def get_selected_features(type, df, dep_vars, model_kwargs):
    model = None

    if type == "decision_tree":
        model = tree.DecisionTreeClassifier(**model_kwargs)
    elif type == "random_forest":
        model = ensemble.RandomForestClassifier(**model_kwargs)
    elif type == "gboost":
        model = ensemble.GradientBoostingClassifier(**model_kwargs)
    else:
        model = tree.ExtraTreeClassifier(**model_kwargs)


    X = df.drop(columns=dep_vars)
    y = df[dep_vars]
    model.fit(X, y)
    imp_df = pd.DataFrame(dict(cols=X.columns, imp=model.feature_importances_))
    imp_df = imp_df[imp_df.imp > 0.005]
    return imp_df.cols.values


def get_sampling_strategy(sampling_strategy):
    if sampling_strategy == "random_over_sampler":
        return RandomOverSampler
    elif sampling_strategy == "smote":
        return SMOTE
    elif sampling_strategy == "svmsmote":
        return SVMSMOTE
    elif sampling_strategy == "adasyn":
        return ADASYN
    elif sampling_strategy == "kmeans_smote":
        return KMeansSMOTE
    else:
        return None



def get_preprocess_pipeline(df, cont_cols, cat_cols, drop_cols):
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

    for feature in cont_cols:
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
    numeric_descr = df.drop(columns=drop_cols).describe().T
    semi_constant_mask = np.isclose(numeric_descr["min"], numeric_descr["50%"])
    semi_constant_descr = numeric_descr[semi_constant_mask]
    semi_const_cols_thresholds = semi_constant_descr["50%"].to_dict()

    # List of columns to be transformed
    semi_const_cols = semi_const_cols_thresholds.keys()
    no_transform_cols = no_transform_cols.drop(semi_const_cols, errors="ignore").to_list()
    log_transform_cols = log_transform_cols.drop(semi_const_cols, errors="ignore").to_list()
    reciprocal_transform_cols = reciprocal_transform_cols.drop(semi_const_cols, errors="ignore").to_list()
    boxcox_transform_cols = boxcox_transform_cols.drop(semi_const_cols, errors="ignore").to_list()
    yeojohnson_transform_cols = yeojohnson_transform_cols.drop(semi_const_cols, errors="ignore").to_list()

    # Transformations
    standard_scaling = (
        preprocessing.StandardScaler(),
        no_transform_cols,
    )
    log_transform = (
        pipeline.make_pipeline(
            preprocessing.FunctionTransformer(func=np.log, feature_names_out="one-to-one"),
            preprocessing.StandardScaler(),
        ),
        log_transform_cols,
    )
    reciprocal_transform = (
        pipeline.make_pipeline(
            preprocessing.FunctionTransformer(func=np.reciprocal, feature_names_out="one-to-one"),
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
            preprocessing.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
        cat_cols,  # type: ignore
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
