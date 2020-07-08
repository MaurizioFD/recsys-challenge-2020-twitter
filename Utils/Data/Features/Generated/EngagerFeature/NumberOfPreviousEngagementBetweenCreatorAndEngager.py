import numpy as np

from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId, MappedFeatureCreatorId
import time


def find_and_increase_engager(eng_id, cre_id, dictionary):
    # Number of time the user_1 has interacted with user_2
    current_count = dictionary.get((cre_id, eng_id), 0)
    dictionary[(cre_id, eng_id)] = current_count + 1
    return current_count

def find_and_increase_creator(eng_id, cre_id, dictionary):
    # Number of time the user_1 has interacted with user_2
    current_count = dictionary.get((eng_id, cre_id), 0)
    dictionary[(cre_id, eng_id)] = dictionary.get((cre_id, eng_id), 0) + 1
    return current_count


class EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByCreator(GeneratedFeaturePickle):

    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_like_engagement_betweet_creator_and_engager_by_creator",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsLike(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_creator(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, cre_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByCreator(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((eng_id, cre_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByCreator(test_dataset_id).save_feature(
                result)


class EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByEngager(GeneratedFeaturePickle):

    # Has the engager ever liked a tweet of the creator? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__(
            "engager_feature_number_of_previous_like_engagement_betweet_creator_and_engager_by_engager",
            dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsLike(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((cre_id, eng_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((cre_id, eng_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).save_feature(result)


class EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByCreator(GeneratedFeaturePickle):

    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_reply_engagement_betweet_creator_and_engager_by_creator",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsReply(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_creator(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, cre_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByCreator(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((eng_id, cre_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByCreator(
                test_dataset_id).save_feature(
                result)


class EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByEngager(GeneratedFeaturePickle):

    # Has the engager ever liked a tweet of the creator? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__(
            "engager_feature_number_of_previous_reply_engagement_betweet_creator_and_engager_by_engager",
            dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsReply(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((cre_id, eng_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((cre_id, eng_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).save_feature(result)


class EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByCreator(GeneratedFeaturePickle):

    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_retweet_engagement_betweet_creator_and_engager_by_creator",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsRetweet(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_creator(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, cre_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByCreator(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((eng_id, cre_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByCreator(
                test_dataset_id).save_feature(
                result)


class EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByEngager(GeneratedFeaturePickle):

    # Has the engager ever liked a tweet of the creator? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__(
            "engager_feature_number_of_previous_retweet_engagement_betweet_creator_and_engager_by_engager",
            dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsRetweet(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((cre_id, eng_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((cre_id, eng_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).save_feature(result)


class EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByCreator(GeneratedFeaturePickle):

    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_comment_engagement_betweet_creator_and_engager_by_creator",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsComment(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_creator(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, cre_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByCreator(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((eng_id, cre_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByCreator(
                test_dataset_id).save_feature(
                result)


class EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByEngager(GeneratedFeaturePickle):

    # Has the engager ever liked a tweet of the creator? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__(
            "engager_feature_number_of_previous_comment_engagement_betweet_creator_and_engager_by_engager",
            dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsComment(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((cre_id, eng_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((cre_id, eng_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).save_feature(result)


class EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByCreator(GeneratedFeaturePickle):

    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_positive_engagement_betweet_creator_and_engager_by_creator",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsPositive(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_creator(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, cre_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByCreator(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((eng_id, cre_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByCreator(
                test_dataset_id).save_feature(
                result)


class EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByEngager(GeneratedFeaturePickle):

    # Has the engager ever liked a tweet of the creator? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__(
            "engager_feature_number_of_previous_positive_engagement_betweet_creator_and_engager_by_engager",
            dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsPositive(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((cre_id, eng_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((cre_id, eng_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).save_feature(result)

class EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByCreator(GeneratedFeaturePickle):

    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_negative_engagement_betweet_creator_and_engager_by_creator",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsNegative(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_creator(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, cre_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByCreator(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByCreator(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((eng_id, cre_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByCreator(
                test_dataset_id).save_feature(
                result)


class EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByEngager(GeneratedFeaturePickle):

    # Has the engager ever liked a tweet of the creator? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__(
            "engager_feature_number_of_previous_negative_engagement_betweet_creator_and_engager_by_engager",
            dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_between_engager_and_creator/{self.feature_name}.csv.gz")

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
        engagers_feature = MappedFeatureEngagerId(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsNegative(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, engagement_dict)
             if engagement
             else engagement_dict.get((cre_id, eng_id), 0)
             for eng_id, cre_id, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByEngager(
                train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create()
            ], axis=1)

            result = pd.DataFrame(
                [engagement_dict.get((cre_id, eng_id), 0)
                 for eng_id, cre_id
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name])],
                index=dataframe.index
            )
            EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByEngager(
                test_dataset_id).save_feature(result)