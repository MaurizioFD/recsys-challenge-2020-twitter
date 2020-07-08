import numpy as np
from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId, MappedFeatureTweetHashtags, MappedFeatureCreatorId, MappedFeatureTweetId
import time
from abc import ABC, abstractmethod


def find_and_increase(creator_id, counter_array):
    current_count = counter_array[creator_id]
    counter_array[creator_id] += 1
    return current_count


class CreatorNumberOfPreviousEngagementReceivedAbstract(GeneratedFeaturePickle, ABC):

    def __init__(self, dataset_id: str):

        super().__init__("creator_feature_number_of_previous_" + self._get_suffix() + "_engagements_received", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creator_features/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creator_features/{self.feature_name}.csv.gz")

    @abstractmethod
    def _get_suffix(self) -> str:
        pass

    @abstractmethod
    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        pass

    @classmethod
    def _save_train_result_if_not_present(cls, result, train_dataset_id):
        if not cls(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            cls(train_dataset_id).save_feature(result)

    @classmethod
    def _exists_test_feature(cls, test_dataset_id):
        return cls(test_dataset_id).has_feature()

    @classmethod
    def _save_test_result(cls, result, test_dataset_id):
        cls(test_dataset_id).save_feature(result)

    def create_feature(self):
        # Check if the dataset id is train or test
        if is_test_or_val_set(self.dataset_id):
            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)
            test_dataset_id = self.dataset_id
        else:
            train_dataset_id = self.dataset_id
            test_dataset_id = get_test_or_val_set_id_from_train(train_dataset_id)

        start_time = time.time()

        # Load features
        creation_timestamps_feature = RawFeatureTweetTimestamp(train_dataset_id)
        creator_id_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = self._get_engagement_feature(train_dataset_id)

        # save column names
        creator_id_col = creator_id_feature.feature_name
        engagement_col = engagement_feature.feature_name

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            creator_id_feature.load_or_create(),
            engagement_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        creator_counter_array = np.zeros(dataframe[creator_id_col].max() + 1, dtype=int)

        result = pd.DataFrame(
            [find_and_increase(creator_id=creator_id, counter_array=creator_counter_array)
             if engagement else creator_counter_array[creator_id]
             for creator_id, engagement in zip(dataframe[creator_id_col], dataframe[engagement_col])],
            index=dataframe.index
        )
        self._save_train_result_if_not_present(result, train_dataset_id)

        if not self._exists_test_feature(test_dataset_id):
            # Load features
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            creator_id_feature = MappedFeatureCreatorId(test_dataset_id)

            # save column names
            creator_id_col = creator_id_feature.feature_name

            dataframe = pd.concat([
                creator_id_feature.load_or_create(),
                creation_timestamps_feature.load_or_create(),
            ], axis=1)

            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

            if dataframe[creator_id_col].max() + 1 > creator_counter_array.size:
                creator_counter_array = np.pad(
                    creator_counter_array,
                    pad_width=(0, dataframe[creator_id_col].max() + 1 - creator_counter_array.size),
                    mode='constant',
                    constant_values=0
                )

            result = pd.DataFrame(dataframe[creator_id_col].map(lambda x: creator_counter_array[x]),
                                  index=dataframe.index)

            result.sort_index(inplace=True)

            print("time:")
            print(time.time() - start_time)

            self._save_test_result(result, test_dataset_id)


class CreatorNumberOfPreviousEngagementReceivedPositive(CreatorNumberOfPreviousEngagementReceivedAbstract):

    def _get_suffix(self) -> str:
        return "positive"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsPositive(dataset_id=dataset_id)


class CreatorNumberOfPreviousEngagementReceivedNegative(CreatorNumberOfPreviousEngagementReceivedAbstract):

    def _get_suffix(self) -> str:
        return "negative"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsNegative(dataset_id=dataset_id)


class CreatorNumberOfPreviousEngagementReceivedLike(CreatorNumberOfPreviousEngagementReceivedAbstract):

    def _get_suffix(self) -> str:
        return "like"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsLike(dataset_id=dataset_id)


class CreatorNumberOfPreviousEngagementReceivedRetweet(CreatorNumberOfPreviousEngagementReceivedAbstract):

    def _get_suffix(self) -> str:
        return "retweet"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsRetweet(dataset_id=dataset_id)


class CreatorNumberOfPreviousEngagementReceivedReply(CreatorNumberOfPreviousEngagementReceivedAbstract):

    def _get_suffix(self) -> str:
        return "reply"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsReply(dataset_id=dataset_id)


class CreatorNumberOfPreviousEngagementReceivedComment(CreatorNumberOfPreviousEngagementReceivedAbstract):

    def _get_suffix(self) -> str:
        return "comment"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsComment(dataset_id=dataset_id)

