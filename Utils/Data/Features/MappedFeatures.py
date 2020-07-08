import pandas as pd
import pathlib as pl
import numpy as np
import RootPath
from abc import abstractmethod
from Utils.Data.Features.RawFeatures import *
from Utils.Data.Dictionary.MappingDictionary import *


def map_column_single_value(series, dictionary):
    mapped_series = series.map(dictionary).astype(np.int32)
    return pd.DataFrame(mapped_series)


def map_column_array(series, dictionary):
    mapped_series = series.map(
        lambda x: np.array([dictionary[y] for y in x.split('\t')], dtype=np.int32) if x is not pd.NA else None)
    return pd.DataFrame(mapped_series)


class MappedFeaturePickle(Feature):
    """
    Abstract class representing a dictionary that works with pickle file.
    """

    def __init__(self, feature_name: str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(f"{Feature.ROOT_PATH}/{self.dataset_id}/mapped/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(f"{Feature.ROOT_PATH}/{self.dataset_id}/mapped/{self.feature_name}.csv.gz")

    def has_feature(self):
        return self.pck_path.is_file()

    def load_feature(self):
        assert self.has_feature(), f"The feature {self.feature_name} does not exists. Create it first."
        df = pd.read_pickle(self.pck_path, compression="gzip")
        # Renaming the column for consistency purpose
        df.columns = [self.feature_name]
        return df

    @abstractmethod
    def create_feature(self):
        pass

    def save_feature(self, dataframe: pd.DataFrame):
        # Changing column name
        dataframe.columns = [self.feature_name]
        self.pck_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_pickle(self.pck_path, compression='gzip')
        # For backup reason
        # self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # dataframe.to_csv(self.csv_path, compression='gzip', index=True)


class MappedFeatureTweetLanguage(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_language", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetLanguage(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingLanguageDictionary().load_or_create()
        mapped_dataframe = map_column_single_value(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureGroupedTweetLanguage(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_grouped_tweet_language", dataset_id)
        
        self.group_id_dict = {}
        self.current_mapping = 0
        
    def get_grouped_id(self, language_id):
        # ??? inglese misto altre cose
        if language_id == 16 or language_id == 18 or language_id == 20:
            return 16
        # [UNK]
        elif language_id == 26 or language_id == 56 or language_id == 57 or language_id == 58 or language_id == 59 or language_id == 61:
            return 26
        # ???
        elif language_id == 28 or language_id == 36 or language_id == 37 or language_id == 43 or language_id == 45 or language_id == 46:
            return 28
        # persian / pashto
        elif language_id == 25 or language_id == 44 or language_id == 41:
            return 25
        # lingue indiane
        elif language_id == 8 or language_id == 32 or language_id == 34 or language_id == 35 or language_id == 47 or language_id == 48 or language_id == 49 or language_id == 50 or language_id == 52 or language_id == 53 or language_id == 54 or language_id == 60 or language_id == 62:
            return 8
        # lingue est europa
        elif language_id == 14 or language_id == 23 or language_id == 24 or language_id == 55:
            return 14
        # lingue nord europa
        elif language_id == 21 or language_id == 31 or language_id == 38 or language_id == 39:
            return 21
        # lingue centro europa / balcani
        elif language_id == 29 or language_id == 40 or language_id == 42:
            return 29
        # others (vietnamita, birmano, armeno, georgiano, uiguro)
        elif language_id == 30 or language_id == 51 or language_id == 63 or language_id == 64 or language_id == 65:
            return 30
        else:
            return language_id
        
    def remap_language_id(self, group_id):
        if group_id not in self.group_id_dict:
            self.group_id_dict[group_id] = self.current_mapping
            self.current_mapping += 1
        return self.group_id_dict[group_id]
        
    def create_feature(self):
        feature = MappedFeatureTweetLanguage(self.dataset_id)
        dataframe = feature.load_or_create()
        
        #dataframe = dataframe.head()
        grouped_dataframe = pd.DataFrame(dataframe["mapped_feature_tweet_language"].map(lambda x: self.get_grouped_id(x)))
        #print(grouped_dataframe)
        mapped_dataframe = pd.DataFrame(dataframe["mapped_feature_tweet_language"].map(lambda x: self.remap_language_id(x)))
        #print(mapped_dataframe)

        self.save_feature(mapped_dataframe)


class MappedFeatureTweetId(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_id", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetId(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingTweetIdDictionary().load_or_create()
        mapped_dataframe = map_column_single_value(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureCreatorId(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_creator_id", dataset_id)

    def create_feature(self):
        feature = RawFeatureCreatorId(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingUserIdDictionary().load_or_create()
        mapped_dataframe = map_column_single_value(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureEngagerId(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_engager_id", dataset_id)

    def create_feature(self):
        feature = RawFeatureEngagerId(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingUserIdDictionary().load_or_create()
        mapped_dataframe = map_column_single_value(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureTweetHashtags(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_hashtags", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetHashtags(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingHashtagDictionary().load_or_create()
        mapped_dataframe = map_column_array(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureTweetLinks(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_links", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetLinks(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingLinkDictionary().load_or_create()
        mapped_dataframe = map_column_array(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureTweetDomains(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_domains", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetDomains(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingDomainDictionary().load_or_create()
        mapped_dataframe = map_column_array(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)


class MappedFeatureTweetMedia(MappedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("mapped_feature_tweet_media", dataset_id)

    def create_feature(self):
        feature = RawFeatureTweetMedia(self.dataset_id)
        dataframe = feature.load_or_create()
        dictionary = MappingMediaDictionary().load_or_create()
        mapped_dataframe = map_column_array(dataframe[feature.feature_name], dictionary)

        self.save_feature(mapped_dataframe)
