from pathlib import Path

import numpy as np
import pandas as pd
from fastai.tabular.core import cont_cat_split
from loguru import logger
from sklearn import model_selection

from config.config import Config

from .utils import get_preprocess_pipeline, get_sampling_strategy


class Preprocessor:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.path = Path(self.config.root)

        self.drop_cols = self.config.drop_cols
        self.dep_vars = self.config.dep_vars

        self.untrainable_cols = self.drop_cols + self.dep_vars

        self.is_submission = self.config.is_submission

    def load_data(self):
        train_df = pd.read_csv(self.path / "train.csv", index_col="Id")
        greeks_df = pd.read_csv(self.path / "greeks.csv", index_col="Id")
        self.train_df = pd.merge(train_df, greeks_df, left_index=True, right_index=True)

        # Drops the dep_vars before splitting categorical and continuous variables
        self.cont_names, self.cat_names = cont_cat_split(self.train_df, dep_var=self.untrainable_cols)

        if self.is_submission:
            test_df = pd.read_csv(self.path / "test.csv", index_col="Id")
            self.test_df = pd.merge(test_df, greeks_df, left_index=True, right_index=True, how="left")
            self.test_df.drop(columns=self.drop_cols, inplace=True)

            # Dummy protection for an empty test dataset
            if np.all(np.isclose(self.test_df.select_dtypes("number").sum(), 0)):
                self.test_df[self.cont_names] += 1e-9

        else:
            random_state = self.config.test.get("random_state", 42)
            test_size = self.config.test.get("test_size", 0.2)

            self.train_df, self.test_df = model_selection.train_test_split(self.train_df, test_size=test_size, random_state=random_state)

    def preliminary_preprocessing(self):
        preprocessor = get_preprocess_pipeline(self.train_df, self.cont_names, self.cat_names, self.untrainable_cols)
        self.X_pre = preprocessor.fit_transform(self.train_df.drop(columns=self.untrainable_cols))
        self.train_df = pd.merge(self.X_pre, self.train_df[self.untrainable_cols], left_index=True, right_index=True)

        if not self.is_submission:
            test_data = preprocessor.transform(self.test_df.drop(columns=self.dep_vars))
            self.test_df = pd.merge(test_data, self.test_df[self.dep_vars], left_index=True, right_index=True)
        else:
            test_data = preprocessor.transform(self.test_df)

    def resample(self):
        sampler = get_sampling_strategy(self.config.sampling_strategy)

        if sampler is None:
            return

        logger.info(f"Resampling data with {self.config.sampling_strategy}")
        sampler_kwargs = self.config.sampling_strategy_kwargs
        X_res, y_res = sampler(**sampler_kwargs).fit_resample(self.X_pre, self.train_df[self.dep_vars])
        self.train_df = pd.merge(X_res, y_res, left_index=True, right_index=True)

    def preprocess(self):
        self.load_data()
        self.preliminary_preprocessing()
        self.resample()

        return self.train_df, self.test_df, self.cont_names, self.cat_names
