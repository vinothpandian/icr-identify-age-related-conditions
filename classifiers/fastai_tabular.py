import numpy as np
import wandb
from fastai.callback.progress import CSVLogger
from fastai.callback.schedule import slide, valley
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.callback.wandb import WandbCallback
from fastai.data.block import CategoryBlock
from fastai.data.transforms import TrainTestSplitter
from fastai.losses import FocalLossFlat
from fastai.metrics import accuracy, skm_to_fastai
from fastai.tabular.core import Categorify
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.learner import tabular_learner
from matplotlib import pyplot as plt
from sklearn import metrics

from .base import BaseClassifier


class FastAITabularClassifier(BaseClassifier):
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
