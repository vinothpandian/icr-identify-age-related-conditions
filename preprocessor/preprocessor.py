from pathlib import Path

import numpy as np
import pandas as pd
from fastai.tabular.core import cont_cat_split

from preprocessor.utils import (
    get_selected_features,
    get_sampling_strategy,
    get_preprocess_pipeline,
)
from loguru import logger


class Preprocessor:
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
        self.cont_names, self.cat_names = cont_cat_split(self.train_df, dep_var=self.dep_vars)
        self.test_df = pd.read_csv(self.path / self.config.test.file, index_col="Id")
        self.test_df.drop(columns=self.drop_vars, inplace=True)

        # Dummy protection for an empty test dataset
        if np.all(np.isclose(self.test_df.select_dtypes("number").sum(), 0)):
            self.test_df[self.cont_names] += 1e-9

    def preliminary_preprocessing(self):
        preprocessor = get_preprocess_pipeline(self.train_df, self.cont_names, self.dep_vars)
        self.X_pre = preprocessor.fit_transform(self.train_df.drop(columns=self.dep_vars))
        self.train_df = pd.merge(self.X_pre, self.train_df[self.dep_vars], left_index=True, right_index=True)
        test_data = preprocessor.transform(self.test_df.drop(columns=self.dep_vars))
        self.test_df = pd.merge(test_data, self.test_df[self.dep_vars], left_index=True, right_index=True)

    def resample(self):
        if sampler := get_sampling_strategy(self.config.train.get("sampling_strategy")):
            logger.info(f"Resampling data with {self.config.train.sampling_strategy}")
            sampler_kwargs = self.config.train.get("sampling_strategy_kwargs", {})
            X_res, y_res = sampler(**sampler_kwargs).fit_resample(self.X_pre, self.train_df[self.dep_vars])
            self.train_df = pd.merge(X_res, y_res, left_index=True, right_index=True)

    def select_features(self):
        if feature_selector := self.config.train.get("feature_selector"):
            selector_kwargs = self.config.train.get("feature_selector_kwargs", {})
            features = get_selected_features(feature_selector, self.train_df, self.dep_vars, selector_kwargs)
            selected_features = [*features, *self.dep_vars]
            self.train_df = self.train_df[selected_features]
            self.test_df = self.test_df[selected_features]

            self.cont_names = [col for col in self.cont_names if col in selected_features]
            self.cat_names = [col for col in self.cat_names if col in selected_features]

    def preprocess(self):
        self.load_data()
        self.preliminary_preprocessing()
        self.resample()
        self.select_features()

        return self.train_df, self.test_df, self.cont_names, self.cat_names
