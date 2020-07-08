import numpy as np

import Utils.Data as data
from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Dictionary.UserBasicFeaturesDictArray import UserBasicFeatureDictArrayNumpy
from Utils.Data.Dictionary.UserFeaturesDictArray import MainLanguageUserBasicFeatureDictArray
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId, MappedFeatureCreatorId, \
    MappedFeatureTweetLanguage, MappedFeatureTweetId, MappedFeatureGroupedTweetLanguage
from Utils.Data.Sparse.CSR_SparseMatrix import CSR_SparseMatrix
import numpy as np
import scipy.sparse as sps
import time
import multiprocessing as mp

def find_and_increase_engager(engager, creator, language, engagement, counter_array):
    if counter_array[engager].sum() < 1:
        current_count = -1
    else:
        current_count = np.argmax(counter_array[engager])
    if engagement:
        counter_array[engager][language] = counter_array[engager][language] + 1
    counter_array[creator][language] = counter_array[creator][language] + 1
    return current_count

def find_and_increase_creator(engager, creator, language, engagement, counter_array):
    if counter_array[creator].sum() < 1:
        current_count = -1
    else:
        current_count = np.argmax(counter_array[creator])
    if engagement:
        counter_array[engager][language] = counter_array[engager][language] + 1
    counter_array[creator][language] = counter_array[creator][language] + 1
    return current_count

def find_and_increase_creator(engager, creator, language, engagement, counter_array):
    if counter_array[creator].sum() < 1:
        return -1
    current_count = np.argmax(counter_array[creator])
    if engagement:
        counter_array[engager][language] = counter_array[engager][language] + 1
    counter_array[creator][language] = counter_array[creator][language] + 1
    return current_count

class EngagerMainGroupedLanguage(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_main_grouped_language", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/grouped_main_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/grouped_main_language/{self.feature_name}.csv.gz")

    def create_feature(self):

        # Check if the dataset id is train or test
        if is_test_or_val_set(self.dataset_id):
            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)
            test_dataset_id = self.dataset_id
        else:
            train_dataset_id = self.dataset_id
            test_dataset_id = get_test_or_val_set_id_from_train(train_dataset_id)

        # Load features
        creation_timestamps_feature = RawFeatureTweetTimestamp(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        language_feature = MappedFeatureGroupedTweetLanguage(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsPositive(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            creators_feature.load_or_create(),
            engagers_feature.load_or_create(),
            language_feature.load_or_create(),
            engagement_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        engager_counter_array = np.zeros((data.DataStats.get_max_user_id() + 1, 70), dtype=np.uint16)

        result = pd.DataFrame(
            [find_and_increase_engager(engager_id, creator_id, language, engagement, engager_counter_array)
             for engager_id, creator_id, language, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[language_feature.feature_name],
                    dataframe[engagement_feature.feature_name]
                    )],
            index=dataframe.index
        )
        if not EngagerMainGroupedLanguage(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerMainGroupedLanguage(train_dataset_id).save_feature(result)
        if not EngagerMainGroupedLanguage(test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            language_feature = MappedFeatureGroupedTweetLanguage(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                creators_feature.load_or_create(),
                engagers_feature.load_or_create(),
                language_feature.load_or_create()
            ], axis=1)

            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

            result = pd.DataFrame(
                [find_and_increase_engager(engager_id, creator_id, language, False, engager_counter_array)
                 for engager_id, creator_id, language
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name],
                        dataframe[language_feature.feature_name]
                        )],
                index=dataframe.index
            )

            result.sort_index(inplace=True)

            EngagerMainGroupedLanguage(test_dataset_id).save_feature(result)


class CreatorMainGroupedLanguage(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("creator_main_grouped_language", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/grouped_main_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/grouped_main_language/{self.feature_name}.csv.gz")

    def create_feature(self):

        # Check if the dataset id is train or test
        if is_test_or_val_set(self.dataset_id):
            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)
            test_dataset_id = self.dataset_id
        else:
            train_dataset_id = self.dataset_id
            test_dataset_id = get_test_or_val_set_id_from_train(train_dataset_id)

        # Load features
        creation_timestamps_feature = RawFeatureTweetTimestamp(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        language_feature = MappedFeatureGroupedTweetLanguage(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsPositive(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            creators_feature.load_or_create(),
            engagers_feature.load_or_create(),
            language_feature.load_or_create(),
            engagement_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        engager_counter_array = np.zeros((data.DataStats.get_max_user_id() + 1, 70), dtype=np.uint16)

        result = pd.DataFrame(
            [find_and_increase_creator(engager_id, creator_id, language, engagement, engager_counter_array)
             for engager_id, creator_id, language, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[language_feature.feature_name],
                    dataframe[engagement_feature.feature_name]
                    )],
            index=dataframe.index
        )
        if not CreatorMainGroupedLanguage(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            CreatorMainGroupedLanguage(train_dataset_id).save_feature(result)
        if not CreatorMainGroupedLanguage(test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            language_feature = MappedFeatureGroupedTweetLanguage(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                creators_feature.load_or_create(),
                engagers_feature.load_or_create(),
                language_feature.load_or_create()
            ], axis=1)

            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

            result = pd.DataFrame(
                [find_and_increase_creator(engager_id, creator_id, language, False, engager_counter_array)
                 for engager_id, creator_id, language
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name],
                        dataframe[language_feature.feature_name]
                        )],
                index=dataframe.index
            )

            result.sort_index(inplace=True)

            CreatorMainGroupedLanguage(test_dataset_id).save_feature(result)


class CreatorAndEngagerHaveSameMainGroupedLanguage(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("creator_and_engager_have_same_main_grouped_language", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/grouped_main_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/grouped_main_language/{self.feature_name}.csv.gz")

    def create_feature(self):
        creator_main_language_feature = CreatorMainGroupedLanguage(self.dataset_id)
        engager_main_language_feature = EngagerMainGroupedLanguage(self.dataset_id)

        creator_main_language_df = creator_main_language_feature.load_or_create()
        engager_main_language_df = engager_main_language_feature.load_or_create()

        result = pd.DataFrame(
            [x == y if x is not -1 else False
             for x, y
             in zip(creator_main_language_df[creator_main_language_feature.feature_name],
                    engager_main_language_df[engager_main_language_feature.feature_name])
             ]
        )

        self.save_feature(result)


class IsTweetInCreatorMainGroupedLanguage(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("is_tweet_in_creator_main_grouped_language", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/grouped_main_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/grouped_main_language/{self.feature_name}.csv.gz")

    def create_feature(self):
        creator_main_language_feature = CreatorMainGroupedLanguage(self.dataset_id)
        tweet_language_feature = MappedFeatureGroupedTweetLanguage(self.dataset_id)

        creator_main_language_df = creator_main_language_feature.load_or_create()
        tweet_language_df = tweet_language_feature.load_or_create()

        result = pd.DataFrame(
            [x == y
             for x, y
             in zip(creator_main_language_df[creator_main_language_feature.feature_name],
                    tweet_language_df[tweet_language_feature.feature_name])]
        )

        self.save_feature(result)


class IsTweetInEngagerMainGroupedLanguage(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("is_tweet_in_engager_main_grouped_language", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/grouped_main_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/grouped_main_language/{self.feature_name}.csv.gz")

    def create_feature(self):
        engager_main_language_feature = EngagerMainGroupedLanguage(self.dataset_id)
        tweet_language_feature = MappedFeatureGroupedTweetLanguage(self.dataset_id)

        engager_main_language_df = engager_main_language_feature.load_or_create()
        tweet_language_df = tweet_language_feature.load_or_create()

        result = pd.DataFrame(
            [x == y
             for x, y
             in zip(engager_main_language_df[engager_main_language_feature.feature_name],
                    tweet_language_df[tweet_language_feature.feature_name])]
        )

        self.save_feature(result)