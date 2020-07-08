import pandas as pd
import pathlib as pl
import numpy as np
import RootPath
from abc import abstractmethod

from Utils.Data.Features.MappedFeatures import *
from Utils.Data.Dictionary.MappingDictionary import *


class TweetBasicFeatureDictArrayNumpy(Dictionary):
    """
    It is built only using train and test set.
    Abstract class representing a dictionary array that works with numpy/pickle file.
    """

    def __init__(self, dictionary_name: str, ):
        super().__init__(dictionary_name)
        self.npz_path = pl.Path(f"{Dictionary.ROOT_PATH}/basic_features/tweet/{self.dictionary_name}.npz")

    def has_dictionary(self):
        return self.npz_path.is_file()

    def load_dictionary(self):
        assert self.has_dictionary(), f"The feature {self.dictionary_name} does not exists. Create it first."
        return np.load(self.npz_path, allow_pickle=True)['x']

    @abstractmethod
    def create_dictionary(self):
        pass

    def save_dictionary(self, arr: np.ndarray):
        self.npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.npz_path, x=arr)


class HashtagsTweetBasicFeatureDictArray(TweetBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("hashtags_tweet_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_tweet_id_feature = MappedFeatureTweetId("train")
        test_tweet_id_feature = MappedFeatureTweetId("test")
        last_test_tweet_id_feature = MappedFeatureTweetId("last_test")

        # Find the mask of uniques one
        train_df = train_tweet_id_feature.load_or_create()
        test_df = test_tweet_id_feature.load_or_create()
        last_test_df = last_test_tweet_id_feature.load_or_create()

        test_df = pd.concat([test_df, last_test_df])

        unique = ~train_df[train_tweet_id_feature.feature_name].append(
            test_df[test_tweet_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = train_df[train_tweet_id_feature.feature_name].append(test_df[test_tweet_id_feature.feature_name])[
            unique]

        # Load the target column
        column = "hashtags"
        train_target_feature = MappedFeatureTweetHashtags("train")
        test_target_feature = MappedFeatureTweetHashtags("test")
        last_test_target_feature = MappedFeatureTweetHashtags("last_test")
        train_df = train_target_feature.load_or_create()
        test_df = test_target_feature.load_or_create()
        last_test_df = last_test_target_feature.load_or_create()
        test_df = pd.concat([test_df, last_test_df])
        df[column] = train_df[train_target_feature.feature_name].append(test_df[test_target_feature.feature_name])[
            unique]
        df[column] = df[column].map(lambda array: [int(x) for x in array] if array is not None else None)

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class MediaTweetBasicFeatureDictArray(TweetBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("media_tweet_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_tweet_id_feature = MappedFeatureTweetId("train")
        test_tweet_id_feature = MappedFeatureTweetId("test")
        last_test_tweet_id_feature = MappedFeatureTweetId("last_test")

        # Find the mask of uniques one
        train_df = train_tweet_id_feature.load_or_create()
        test_df = test_tweet_id_feature.load_or_create()
        last_test_df = last_test_tweet_id_feature.load_or_create()

        test_df = pd.concat([test_df, last_test_df])

        unique = ~train_df[train_tweet_id_feature.feature_name].append(
            test_df[test_tweet_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = train_df[train_tweet_id_feature.feature_name].append(test_df[test_tweet_id_feature.feature_name])[
            unique]

        # Load the target column
        column = "media"
        train_target_feature = MappedFeatureTweetMedia("train")
        test_target_feature = MappedFeatureTweetMedia("test")
        last_test_target_feature = MappedFeatureTweetMedia("last_test")
        train_df = train_target_feature.load_or_create()
        test_df = test_target_feature.load_or_create()
        last_test_df = last_test_target_feature.load_or_create()
        test_df = pd.concat([test_df, last_test_df])
        df[column] = train_df[train_target_feature.feature_name].append(test_df[test_target_feature.feature_name])[
            unique]
        df[column] = df[column].map(lambda array: [int(x) for x in array] if array is not None else None)

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class LinksTweetBasicFeatureDictArray(TweetBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("links_tweet_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_tweet_id_feature = MappedFeatureTweetId("train")
        test_tweet_id_feature = MappedFeatureTweetId("test")
        last_test_tweet_id_feature = MappedFeatureTweetId("last_test")

        # Find the mask of uniques one
        train_df = train_tweet_id_feature.load_or_create()
        test_df = test_tweet_id_feature.load_or_create()
        last_test_df = last_test_tweet_id_feature.load_or_create()

        test_df = pd.concat([test_df, last_test_df])

        unique = ~train_df[train_tweet_id_feature.feature_name].append(
            test_df[test_tweet_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = train_df[train_tweet_id_feature.feature_name].append(test_df[test_tweet_id_feature.feature_name])[
            unique]

        # Load the target column
        column = "links"
        train_target_feature = MappedFeatureTweetLinks("train")
        test_target_feature = MappedFeatureTweetLinks("test")
        last_test_target_feature = MappedFeatureTweetLinks("last_test")
        train_df = train_target_feature.load_or_create()
        test_df = test_target_feature.load_or_create()
        last_test_df = last_test_target_feature.load_or_create()
        test_df = pd.concat([test_df, last_test_df])
        df[column] = train_df[train_target_feature.feature_name].append(test_df[test_target_feature.feature_name])[
            unique]
        df[column] = df[column].map(lambda array: [int(x) for x in array] if array is not None else None)

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class DomainsTweetBasicFeatureDictArray(TweetBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("domains_tweet_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_tweet_id_feature = MappedFeatureTweetId("train")
        test_tweet_id_feature = MappedFeatureTweetId("test")
        last_test_tweet_id_feature = MappedFeatureTweetId("last_test")

        # Find the mask of uniques one
        train_df = train_tweet_id_feature.load_or_create()
        test_df = test_tweet_id_feature.load_or_create()
        last_test_df = last_test_tweet_id_feature.load_or_create()

        test_df = pd.concat([test_df, last_test_df])

        unique = ~train_df[train_tweet_id_feature.feature_name].append(
            test_df[test_tweet_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = train_df[train_tweet_id_feature.feature_name].append(test_df[test_tweet_id_feature.feature_name])[
            unique]

        # Load the target column
        column = "domains"
        train_target_feature = MappedFeatureTweetDomains("train")
        test_target_feature = MappedFeatureTweetDomains("test")
        last_test_target_feature = MappedFeatureTweetDomains("last_test")
        train_df = train_target_feature.load_or_create()
        test_df = test_target_feature.load_or_create()
        last_test_df = last_test_target_feature.load_or_create()
        test_df = pd.concat([test_df, last_test_df])
        df[column] = train_df[train_target_feature.feature_name].append(test_df[test_target_feature.feature_name])[
            unique]
        df[column] = df[column].map(lambda array: [int(x) for x in array] if array is not None else None)

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class TypeTweetBasicFeatureDictArray(TweetBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("type_tweet_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_tweet_id_feature = MappedFeatureTweetId("train")
        test_tweet_id_feature = MappedFeatureTweetId("test")
        last_test_tweet_id_feature = MappedFeatureTweetId("last_test")

        # Find the mask of uniques one
        train_df = train_tweet_id_feature.load_or_create()
        test_df = test_tweet_id_feature.load_or_create()
        last_test_df = last_test_tweet_id_feature.load_or_create()

        test_df = pd.concat([test_df, last_test_df])

        unique = ~train_df[train_tweet_id_feature.feature_name].append(
            test_df[test_tweet_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = train_df[train_tweet_id_feature.feature_name].append(test_df[test_tweet_id_feature.feature_name])[
            unique]

        # Load the target column
        column = "type"
        train_target_feature = RawFeatureTweetType("train")
        test_target_feature = RawFeatureTweetType("test")
        last_test_target_feature = RawFeatureTweetType("last_test")
        train_df = train_target_feature.load_or_create()
        test_df = test_target_feature.load_or_create()
        last_test_df = last_test_target_feature.load_or_create()
        test_df = pd.concat([test_df, last_test_df])
        df[column] = train_df[train_target_feature.feature_name].append(test_df[test_target_feature.feature_name])[
            unique]

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class TimestampTweetBasicFeatureDictArray(TweetBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("timestamp_tweet_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_tweet_id_feature = MappedFeatureTweetId("train")
        test_tweet_id_feature = MappedFeatureTweetId("test")
        last_test_tweet_id_feature = MappedFeatureTweetId("last_test")

        # Find the mask of uniques one
        train_df = train_tweet_id_feature.load_or_create()
        test_df = test_tweet_id_feature.load_or_create()
        last_test_df = last_test_tweet_id_feature.load_or_create()

        test_df = pd.concat([test_df, last_test_df])

        unique = ~train_df[train_tweet_id_feature.feature_name].append(
            test_df[test_tweet_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = train_df[train_tweet_id_feature.feature_name].append(test_df[test_tweet_id_feature.feature_name])[
            unique]

        # Load the target column
        column = "timestamp"
        train_target_feature = RawFeatureTweetTimestamp("train")
        test_target_feature = RawFeatureTweetTimestamp("test")
        last_test_target_feature = RawFeatureTweetTimestamp("last_test")
        train_df = train_target_feature.load_or_create()
        test_df = test_target_feature.load_or_create()
        last_test_df = last_test_target_feature.load_or_create()
        test_df = pd.concat([test_df, last_test_df])
        df[column] = train_df[train_target_feature.feature_name].append(test_df[test_target_feature.feature_name])[
            unique]

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class CreatorIdTweetBasicFeatureDictArray(TweetBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("creator_id_tweet_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_tweet_id_feature = MappedFeatureTweetId("train")
        test_tweet_id_feature = MappedFeatureTweetId("test")
        last_test_tweet_id_feature = MappedFeatureTweetId("last_test")

        # Find the mask of uniques one
        train_df = train_tweet_id_feature.load_or_create()
        test_df = test_tweet_id_feature.load_or_create()
        last_test_df = last_test_tweet_id_feature.load_or_create()

        test_df = pd.concat([test_df, last_test_df])

        unique = ~train_df[train_tweet_id_feature.feature_name].append(
            test_df[test_tweet_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = train_df[train_tweet_id_feature.feature_name].append(test_df[test_tweet_id_feature.feature_name])[
            unique]

        # Load the target column
        column = "creator_id"
        train_target_feature = MappedFeatureCreatorId("train")
        test_target_feature = MappedFeatureCreatorId("test")
        last_test_target_feature = MappedFeatureCreatorId("last_test")
        train_df = train_target_feature.load_or_create()
        test_df = test_target_feature.load_or_create()
        last_test_df = last_test_target_feature.load_or_create()
        test_df = pd.concat([test_df, last_test_df])
        df[column] = train_df[train_target_feature.feature_name].append(test_df[test_target_feature.feature_name])[
            unique]

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)

