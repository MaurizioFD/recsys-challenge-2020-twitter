import numpy as np

from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId, MappedFeatureCreatorId, \
    MappedFeatureTweetLanguage
import time


def find_and_increase_engager(eng_id, cre_id, lang, dictionary):
    # Number of time the user_1 has interacted with user_2
    current_count = dictionary.get((eng_id, lang), 0)
    dictionary[(cre_id, lang)] = dictionary.get((cre_id, lang), 0) + 1
    dictionary[(eng_id, lang)] = current_count + 1
    return current_count

def find_and_increase_creator(eng_id, cre_id, lang, dictionary):
    # Number of time the user_1 has interacted with user_2
    current_count = dictionary.get((eng_id, lang), 0)
    dictionary[(cre_id, lang)] = dictionary.get((cre_id, lang), 0) + 1
    return current_count


class EngagerFeatureNumberOfPreviousLikeEngagementWithLanguage(GeneratedFeaturePickle):
    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_like_engagement_with_language",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.csv.gz")
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
        language_feature = MappedFeatureTweetLanguage(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsLike(train_dataset_id)
        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create(),
            language_feature.load_or_create()
        ], axis=1)
        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)
        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}
        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, lang, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, lang), 0)
             for eng_id, cre_id, lang, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[language_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )
        if not EngagerFeatureNumberOfPreviousLikeEngagementWithLanguage(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousLikeEngagementWithLanguage(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousLikeEngagementWithLanguage(test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            language_feature = MappedFeatureTweetLanguage(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)
            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create(),
                language_feature.load_or_create()
            ], axis=1)
            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)
            result = pd.DataFrame(
                [find_and_increase_creator(eng_id, cre_id, lang, engagement_dict)
                 for eng_id, cre_id, lang
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name],
                        dataframe[language_feature.feature_name])],
                index=dataframe.index
            )
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousLikeEngagementWithLanguage(test_dataset_id).save_feature(result)

class EngagerFeatureNumberOfPreviousRetweetEngagementWithLanguage(GeneratedFeaturePickle):

    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_retweet_engagement_with_language",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.csv.gz")

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
        language_feature = MappedFeatureTweetLanguage(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsRetweet(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create(),
            language_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, lang, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, lang), 0)
             for eng_id, cre_id, lang, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[language_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousRetweetEngagementWithLanguage(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousRetweetEngagementWithLanguage(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousRetweetEngagementWithLanguage(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            language_feature = MappedFeatureTweetLanguage(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create(),
                language_feature.load_or_create()
            ], axis=1)

            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

            result = pd.DataFrame(
                [find_and_increase_creator(eng_id, cre_id, lang, engagement_dict)
                 for eng_id, cre_id, lang
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name],
                        dataframe[language_feature.feature_name])],
                index=dataframe.index
            )
            result.sort_index(inplace=True)

            EngagerFeatureNumberOfPreviousRetweetEngagementWithLanguage(test_dataset_id).save_feature(result)

class EngagerFeatureNumberOfPreviousReplyEngagementWithLanguage(GeneratedFeaturePickle):

    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_reply_engagement_with_language",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.csv.gz")

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
        language_feature = MappedFeatureTweetLanguage(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsReply(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create(),
            language_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, lang, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, lang), 0)
             for eng_id, cre_id, lang, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[language_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousReplyEngagementWithLanguage(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousReplyEngagementWithLanguage(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousReplyEngagementWithLanguage(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            language_feature = MappedFeatureTweetLanguage(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create(),
                language_feature.load_or_create()
            ], axis=1)

            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

            result = pd.DataFrame(
                [find_and_increase_creator(eng_id, cre_id, lang, engagement_dict)
                 for eng_id, cre_id, lang
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name],
                        dataframe[language_feature.feature_name])],
                index=dataframe.index
            )
            result.sort_index(inplace=True)

            EngagerFeatureNumberOfPreviousReplyEngagementWithLanguage(test_dataset_id).save_feature(result)


class EngagerFeatureNumberOfPreviousCommentEngagementWithLanguage(GeneratedFeaturePickle):

    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_comment_engagement_with_language",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.csv.gz")

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
        language_feature = MappedFeatureTweetLanguage(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsComment(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create(),
            language_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, lang, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, lang), 0)
             for eng_id, cre_id, lang, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[language_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousCommentEngagementWithLanguage(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousCommentEngagementWithLanguage(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousCommentEngagementWithLanguage(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            language_feature = MappedFeatureTweetLanguage(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create(),
                language_feature.load_or_create()
            ], axis=1)

            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

            result = pd.DataFrame(
                [find_and_increase_creator(eng_id, cre_id, lang, engagement_dict)
                 for eng_id, cre_id, lang
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name],
                        dataframe[language_feature.feature_name])],
                index=dataframe.index
            )
            result.sort_index(inplace=True)

            EngagerFeatureNumberOfPreviousCommentEngagementWithLanguage(test_dataset_id).save_feature(result)

class EngagerFeatureNumberOfPreviousPositiveEngagementWithLanguage(GeneratedFeaturePickle):

    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_positive_engagement_with_language",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.csv.gz")

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
        language_feature = MappedFeatureTweetLanguage(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsPositive(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create(),
            language_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, lang, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, lang), 0)
             for eng_id, cre_id, lang, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[language_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousPositiveEngagementWithLanguage(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousPositiveEngagementWithLanguage(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousPositiveEngagementWithLanguage(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            language_feature = MappedFeatureTweetLanguage(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create(),
                language_feature.load_or_create()
            ], axis=1)

            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

            result = pd.DataFrame(
                [find_and_increase_creator(eng_id, cre_id, lang, engagement_dict)
                 for eng_id, cre_id, lang
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name],
                        dataframe[language_feature.feature_name])],
                index=dataframe.index
            )
            result.sort_index(inplace=True)

            EngagerFeatureNumberOfPreviousPositiveEngagementWithLanguage(test_dataset_id).save_feature(result)

class EngagerFeatureNumberOfPreviousNegativeEngagementWithLanguage(GeneratedFeaturePickle):

    # Has the creator ever liked a tweet of the engager? If yes, how many times?
    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_negative_engagement_with_language",
                         dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagement_with_language/{self.feature_name}.csv.gz")

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
        language_feature = MappedFeatureTweetLanguage(train_dataset_id)
        engagement_feature = TweetFeatureEngagementIsNegative(train_dataset_id)

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create(),
            creators_feature.load_or_create(),
            language_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        # KEY: a tuple (creator, engager)
        # VALUE: the number of time the engager has engaged with the creator
        # If key does not exists -> 0 times.
        engagement_dict = {}

        result = pd.DataFrame(
            [find_and_increase_engager(eng_id, cre_id, lang, engagement_dict)
             if engagement
             else engagement_dict.get((eng_id, lang), 0)
             for eng_id, cre_id, lang, engagement
             in zip(dataframe[engagers_feature.feature_name],
                    dataframe[creators_feature.feature_name],
                    dataframe[language_feature.feature_name],
                    dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousNegativeEngagementWithLanguage(
                train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousNegativeEngagementWithLanguage(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousNegativeEngagementWithLanguage(
                test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)
            language_feature = MappedFeatureTweetLanguage(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
                creators_feature.load_or_create(),
                language_feature.load_or_create()
            ], axis=1)

            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

            result = pd.DataFrame(
                [find_and_increase_creator(eng_id, cre_id, lang, engagement_dict)
                 for eng_id, cre_id, lang
                 in zip(dataframe[engagers_feature.feature_name],
                        dataframe[creators_feature.feature_name],
                        dataframe[language_feature.feature_name])],
                index=dataframe.index
            )
            result.sort_index(inplace=True)

            EngagerFeatureNumberOfPreviousNegativeEngagementWithLanguage(test_dataset_id).save_feature(result)
