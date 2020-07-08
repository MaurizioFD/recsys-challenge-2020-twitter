from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
import pandas as pd
import pathlib as pl
from Utils.Data.Features.Feature import Feature
from abc import ABC
from abc import abstractmethod


class EnsemblingFeatureAbstract(GeneratedFeaturePickle, ABC):

    def __init__(self, df_train: pd.DataFrame, df_train_label: pd.DataFrame,
                 df_to_predict: pd.DataFrame, param_dict: dict):
        # TODO what is the dataset_id??
        dataset_id = self._get_dataset_id()
        feature_name = self._get_feature_name()

        super().__init__(feature_name, dataset_id=dataset_id)

        path = self._get_path()
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/ensembling/{path}/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/ensembling/{path}/{self.feature_name}.csv.gz")

        self.df_train = df_train
        self.df_train_label = df_train_label
        self.df_to_predict = df_to_predict
        self.model_path = f"{Feature.ROOT_PATH}/{self.dataset_id}/ensembling/{path}/{self.feature_name}_model.model"
        self.param_dict = param_dict

    def _get_model(self):
        if not pl.Path(self.model_path).exists():
            self._train_and_save()
        return self._load_model()

    @abstractmethod
    def _get_dataset_id(self):
        pass

    @abstractmethod
    def _get_path(self):
        pass

    @abstractmethod
    def _get_feature_name(self):
        pass

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _train_and_save(self):
        pass

