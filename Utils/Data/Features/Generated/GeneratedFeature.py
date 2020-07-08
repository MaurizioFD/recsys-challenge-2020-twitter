import pandas as pd
import pathlib as pl
import numpy as np
import RootPath
from abc import abstractmethod
from Utils.Data.Features.RawFeatures import *
from Utils.Data.Dictionary.MappingDictionary import *


class GeneratedFeaturePickle(Feature):
    """
    Abstract class representing a generated feature that works with pickle file.
    """

    def __init__(self, feature_name: str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path("")
        self.csv_path = pl.Path("")

    def has_feature(self):
        return self.pck_path.is_file()

    def load_feature(self):
        assert self.has_feature(), f"The feature {self.feature_name} does not exists. Create it first."
        dataframe = pd.read_pickle(self.pck_path, compression="gzip")
        if len(dataframe.columns) == 1:
            # Renaming the column for consistency purpose
            dataframe.columns = [self.feature_name]
        return dataframe

    @abstractmethod
    def create_feature(self):
        pass

    def save_feature(self, dataframe: pd.DataFrame):
        if len(dataframe.columns) == 1:
            # Changing column name
            dataframe.columns = [self.feature_name]
        self.pck_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_pickle(self.pck_path, compression='gzip', protocol=4)
        # For backup reason
        # self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # dataframe.to_csv(self.csv_path, compression='gzip', index=True)

class GeneratedFeatureOnlyPickle(Feature):
    """
    Abstract class representing a generated feature that works with pickle file.
    """

    def __init__(self, feature_name: str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path("")
        self.csv_path = pl.Path("")

    def has_feature(self):
        return self.pck_path.is_file()

    def load_feature(self):
        assert self.has_feature(), f"The feature {self.feature_name} does not exists. Create it first."
        dataframe = pd.read_pickle(self.pck_path, compression="gzip")
        if len(dataframe.columns) == 1:
            # Renaming the column for consistency purpose
            dataframe.columns = [self.feature_name]
        return dataframe

    @abstractmethod
    def create_feature(self):
        pass

    def save_feature(self, dataframe: pd.DataFrame):
        if len(dataframe.columns) == 1:
            # Changing column name
            dataframe.columns = [self.feature_name]
        self.pck_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_pickle(self.pck_path, compression='gzip')