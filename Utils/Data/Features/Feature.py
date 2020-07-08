from abc import abstractmethod
from abc import ABC

import RootPath


class Feature(ABC):
    ROOT_PATH = RootPath.get_root().joinpath("Dataset/Features/")

    def __init__(self, feature_name: str, dataset_id: str):
        self.feature_name = feature_name
        self.dataset_id = dataset_id

    @abstractmethod
    def has_feature(self):
        pass

    @abstractmethod
    def load_feature(self):
        pass

    @abstractmethod
    def create_feature(self):
        pass

    def load_or_create(self):
        if not self.has_feature():
            self.create_feature()
        return self.load_feature()
