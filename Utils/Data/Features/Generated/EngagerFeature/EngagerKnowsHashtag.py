import numpy as np

from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId, MappedFeatureTweetHashtags
import time
from abc import ABC, abstractmethod


class EngagerKnowsHashtagAbstract(GeneratedFeaturePickle, ABC):

    def __init__(self, dataset_id: str):

        super().__init__("engager_feature_knows_hashtag_" + self._get_suffix(), dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engager_features/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engager_features/{self.feature_name}.csv.gz")

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

    def find_similarity_and_update(self, engager_id, hashtag_arr, prev_hashtags_dictionary: dict,
                                   already_seen_users_set: set, is_positive_engagement: bool):
        # if there is no hashtag in the current tweet
        if hashtag_arr is None:
            return 0

        # check if engager_id was already seen
        if engager_id in already_seen_users_set:
            # then, we retrieve the list of its previously engaged tweets
            prev_hashtags_set = prev_hashtags_dictionary[engager_id]

            # drop duplicates
            curr_hashtags_set = set(hashtag_arr)

            # count how many hashtag the user 'engager_id' has already seen
            sim = 0
            # iterate over the array
            # update the prev_hashtags_set if positive engagement

            if is_positive_engagement:
                for curr_hashtag in curr_hashtags_set:
                    if curr_hashtag in prev_hashtags_set:
                        sim += 1
                    else:  # UPDATE only if it wasn't present
                        prev_hashtags_dictionary[engager_id].add(curr_hashtag)
            else:         # NO UPDATE
                for curr_hashtag in curr_hashtags_set:
                    if curr_hashtag in prev_hashtags_set:
                        sim += 1

        else:
            # if the user was never seen, the similarity is 0
            sim = 0
            # update only if positive
            if is_positive_engagement:
                # update the already seen set
                already_seen_users_set.add(engager_id)
                # add the hashtags to the dictionary
                prev_hashtags_dictionary[engager_id] = set(hashtag_arr)

        return sim

    # for the test dataset
    def find_similarity_no_update(self, engager_id, hashtag_arr, prev_hashtags_dictionary: dict,
                                   already_seen_users_set: set):

        # if there is no hashtag in the current tweet
        if hashtag_arr is None:
            return 0

        # check if engager_id was already seen
        if engager_id in already_seen_users_set:
            # then, we retrieve the list of its previously engaged tweets
            prev_hashtags_set = prev_hashtags_dictionary[engager_id]

            # drop duplicates
            curr_hashtags_set = set(hashtag_arr)

            # count how many hashtag the user 'engager_id' has already seen
            sim = 0
            # iterate over the array
            for curr_hashtag in curr_hashtags_set:
                if curr_hashtag in prev_hashtags_set:
                    sim += 1

        else:
            # if the user was never seen, the similarity is 0
            sim = 0

        return sim


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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        hashtag_feature = MappedFeatureTweetHashtags(train_dataset_id)
        engagement_feature = self._get_engagement_feature(train_dataset_id)

        # save column names
        engagers_col = engagers_feature.feature_name
        hashtag_col = hashtag_feature.feature_name
        engagement_col = engagement_feature.feature_name

        # a dictionary that returns for a certain user_id the hashtag ids of the tweets
        # previously engaged by user_id
        # user_id -> set of hashtag ids
        prev_hashtags_dictionary = {}

        # a set that contains all the user_ids of users already seen
        already_seen_users_set = set()

        dataframe = pd.concat([
            engagers_feature.load_or_create(),
            creation_timestamps_feature.load_or_create(),
            hashtag_feature.load_or_create(),
            engagement_feature.load_or_create(),
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        result = pd.DataFrame(
            [self.find_similarity_and_update(engager_id, hashtag_arr, prev_hashtags_dictionary,
                                             already_seen_users_set, is_positive_engagement)
             for engager_id, hashtag_arr, is_positive_engagement in zip(dataframe[engagers_col], dataframe[hashtag_col],
                                                                        dataframe[engagement_col])],
            index=dataframe.index
        )

        self._save_train_result_if_not_present(result, train_dataset_id)
        if not self._exists_test_feature(test_dataset_id):
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            hashtag_feature = MappedFeatureTweetHashtags(test_dataset_id)

            # save column names
            engagers_col = engagers_feature.feature_name
            hashtag_col = hashtag_feature.feature_name

            dataframe = pd.concat([
                engagers_feature.load_or_create(),
                creation_timestamps_feature.load_or_create(),
                hashtag_feature.load_or_create()
            ], axis=1)

            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)
            result = pd.DataFrame(
                [self.find_similarity_no_update(engager_id, hashtag_arr, prev_hashtags_dictionary,
                                                already_seen_users_set)
                for engager_id, hashtag_arr in zip(dataframe[engagers_col], dataframe[hashtag_col])],
                index=dataframe.index
            )

            result.sort_index(inplace=True)

            print("time:")
            print(time.time() - start_time)

            self._save_test_result(result, test_dataset_id)


class EngagerKnowsHashtagPositive(EngagerKnowsHashtagAbstract):

    def _get_suffix(self) -> str:
        return "positive"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsPositive(dataset_id=dataset_id)



class EngagerKnowsHashtagNegative(EngagerKnowsHashtagAbstract):

    def _get_suffix(self) -> str:
        return "negative"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsNegative(dataset_id=dataset_id)


class EngagerKnowsHashtagLike(EngagerKnowsHashtagAbstract):

    def _get_suffix(self) -> str:
        return "like"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsLike(dataset_id=dataset_id)


class EngagerKnowsHashtagRetweet(EngagerKnowsHashtagAbstract):

    def _get_suffix(self) -> str:
        return "retweet"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsRetweet(dataset_id=dataset_id)


class EngagerKnowsHashtagReply(EngagerKnowsHashtagAbstract):

    def _get_suffix(self) -> str:
        return "reply"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsReply(dataset_id=dataset_id)


class EngagerKnowsHashtagComment(EngagerKnowsHashtagAbstract):

    def _get_suffix(self) -> str:
        return "comment"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsComment(dataset_id=dataset_id)

