from abc import abstractmethod
from abc import ABC

import RootPath


class Dictionary(ABC):
    ROOT_PATH = RootPath.get_root().joinpath("Dataset/Dictionary/")

    def __init__(self, dictionary_name: str):
        self.dictionary_name = dictionary_name

    @abstractmethod
    def has_dictionary(self):
        pass

    @abstractmethod
    def load_dictionary(self):
        pass

    @abstractmethod
    def create_dictionary(self):
        pass

    def load_or_create(self):
        if not self.has_dictionary():
            self.create_dictionary()
        return self.load_dictionary()
