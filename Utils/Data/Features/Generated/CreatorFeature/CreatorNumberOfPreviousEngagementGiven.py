import numpy as np
from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId, MappedFeatureTweetHashtags, MappedFeatureCreatorId, MappedFeatureTweetId
import time
from abc import ABC, abstractmethod


def find_and_increase(creator_id, engager_id, counter_array):
    current_count = counter_array[creator_id]
    counter_array[engager_id] += 1
    return current_count


class CreatorNumberOfPreviousEngagementGivenAbstract(GeneratedFeaturePickle, ABC):

    def __init__(self, dataset_id: str):

        super().__init__("creator_feature_number_of_previous_" + self._get_suffix() + "_engagements_given", dataset_id)
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
        engager_id_feature = MappedFeatureEngagerId(train_dataset_id)
        engagement_feature = self._get_engagement_feature(train_dataset_id)

        # save column names
        creator_id_col = creator_id_feature.feature_name
        engager_id_col = engager_id_feature.feature_name
        engagement_col = engagement_feature.feature_name

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            creator_id_feature.load_or_create(),
            engager_id_feature.load_or_create(),
            engagement_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        max_id = max(dataframe[creator_id_col].max(), dataframe[engager_id_col].max())
        counter_array = np.zeros(max_id + 1, dtype=int)

        result = pd.DataFrame(
            [find_and_increase(creator_id=creator_id, engager_id=engager_id, counter_array=counter_array)
             if engagement else counter_array[creator_id]
             for creator_id, engager_id, engagement in
             zip(dataframe[creator_id_col], dataframe[engager_id_col], dataframe[engagement_col])],
            index=dataframe.index
        )
        self._save_train_result_if_not_present(result, train_dataset_id)

        if not self._exists_test_feature(test_dataset_id):
            # Load features
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            creator_id_feature = MappedFeatureCreatorId(test_dataset_id)
            engager_id_feature = MappedFeatureEngagerId(test_dataset_id)

            # save column names
            creator_id_col = creator_id_feature.feature_name
            engager_id_col = engager_id_feature.feature_name

            dataframe = pd.concat([
                creator_id_feature.load_or_create(),
                engager_id_feature.load_or_create(),
                creation_timestamps_feature.load_or_create(),
            ], axis=1)

            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

            max_id = max(dataframe[creator_id_col].max(), dataframe[engager_id_col].max())

            if max_id + 1 > counter_array.size:
                counter_array = np.pad(
                    counter_array,
                    pad_width=(0, max_id + 1 - counter_array.size),
                    mode='constant',
                    constant_values=0
                )

            result = pd.DataFrame(dataframe[creator_id_col].map(lambda x: counter_array[x]),
                                  index=dataframe.index)

            result.sort_index(inplace=True)

            print("time:")
            print(time.time() - start_time)

            self._save_test_result(result, test_dataset_id)


class CreatorNumberOfPreviousEngagementGivenPositive(CreatorNumberOfPreviousEngagementGivenAbstract):

    def _get_suffix(self) -> str:
        return "positive"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsPositive(dataset_id=dataset_id)


class CreatorNumberOfPreviousEngagementGivenNegative(CreatorNumberOfPreviousEngagementGivenAbstract):

    def _get_suffix(self) -> str:
        return "negative"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsNegative(dataset_id=dataset_id)


class CreatorNumberOfPreviousEngagementGivenLike(CreatorNumberOfPreviousEngagementGivenAbstract):

    def _get_suffix(self) -> str:
        return "like"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsLike(dataset_id=dataset_id)


class CreatorNumberOfPreviousEngagementGivenRetweet(CreatorNumberOfPreviousEngagementGivenAbstract):

    def _get_suffix(self) -> str:
        return "retweet"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsRetweet(dataset_id=dataset_id)


class CreatorNumberOfPreviousEngagementGivenReply(CreatorNumberOfPreviousEngagementGivenAbstract):

    def _get_suffix(self) -> str:
        return "reply"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsReply(dataset_id=dataset_id)


class CreatorNumberOfPreviousEngagementGivenComment(CreatorNumberOfPreviousEngagementGivenAbstract):

    def _get_suffix(self) -> str:
        return "comment"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsComment(dataset_id=dataset_id)

