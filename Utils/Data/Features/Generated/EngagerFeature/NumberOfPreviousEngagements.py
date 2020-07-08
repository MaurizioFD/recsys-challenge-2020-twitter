import numpy as np

from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId
import time


def find_and_increase(engager_id, counter_array):
    current_count = counter_array[engager_id]
    counter_array[engager_id] = current_count + 1
    return current_count


class EngagerFeatureNumberOfPreviousLikeEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_like_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.csv.gz")

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
        engagement_feature = TweetFeatureEngagementIsLike(train_dataset_id)

        # Save the column name
        eng_col = engagers_feature.feature_name
        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)
        engager_counter_array = np.zeros(dataframe[engagers_feature.feature_name].max() + 1, dtype=int)

        result = pd.DataFrame(
            [find_and_increase(engager_id, engager_counter_array) if engagement else engager_counter_array[engager_id]
             for engager_id, engagement in zip(dataframe[eng_col], dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )
        if not EngagerFeatureNumberOfPreviousLikeEngagement(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousLikeEngagement(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousLikeEngagement(test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
            ], axis=1)

            if dataframe[engagers_feature.feature_name].max() + 1 > engager_counter_array.size:
                engager_counter_array = np.pad(
                    engager_counter_array,
                    pad_width=(0, dataframe[engagers_feature.feature_name].max() + 1 - engager_counter_array.size),
                    mode='constant',
                    constant_values=0
                )
            result = pd.DataFrame(dataframe[eng_col].map(lambda x: engager_counter_array[x]),
                                  index=dataframe.index)

            EngagerFeatureNumberOfPreviousLikeEngagement(test_dataset_id).save_feature(result)


class EngagerFeatureNumberOfPreviousReplyEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_reply_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.csv.gz")

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
        engagement_feature = TweetFeatureEngagementIsReply(train_dataset_id)

        # Save the column name
        eng_col = engagers_feature.feature_name

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        engager_counter_array = np.zeros(dataframe[engagers_feature.feature_name].max() + 1, dtype=int)

        result = pd.DataFrame(
            [find_and_increase(engager_id, engager_counter_array) if engagement else engager_counter_array[engager_id]
             for engager_id, engagement in zip(dataframe[eng_col], dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousReplyEngagement(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousReplyEngagement(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousReplyEngagement(test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
            ], axis=1)

            if dataframe[engagers_feature.feature_name].max() + 1 > engager_counter_array.size:
                engager_counter_array = np.pad(
                    engager_counter_array,
                    pad_width=(0, dataframe[engagers_feature.feature_name].max() + 1 - engager_counter_array.size),
                    mode='constant',
                    constant_values=0
                )


            result = pd.DataFrame(dataframe[eng_col].map(lambda x: engager_counter_array[x]),
                                  index=dataframe.index)
            EngagerFeatureNumberOfPreviousReplyEngagement(test_dataset_id).save_feature(result)


class EngagerFeatureNumberOfPreviousRetweetEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_retweet_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.csv.gz")

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
        engagement_feature = TweetFeatureEngagementIsRetweet(train_dataset_id)

        # Save the column name
        eng_col = engagers_feature.feature_name

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        engager_counter_array = np.zeros(dataframe[engagers_feature.feature_name].max() + 1, dtype=int)

        result = pd.DataFrame(
            [find_and_increase(engager_id, engager_counter_array) if engagement else engager_counter_array[engager_id]
             for engager_id, engagement in zip(dataframe[eng_col], dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousRetweetEngagement(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousRetweetEngagement(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousRetweetEngagement(test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
            ], axis=1)

            if dataframe[engagers_feature.feature_name].max() + 1 > engager_counter_array.size:
                engager_counter_array = np.pad(
                    engager_counter_array,
                    pad_width=(0, dataframe[engagers_feature.feature_name].max() + 1 - engager_counter_array.size),
                    mode='constant',
                    constant_values=0
                )

            result = pd.DataFrame(dataframe[eng_col].map(lambda x: engager_counter_array[x]),
                                  index=dataframe.index)
            EngagerFeatureNumberOfPreviousRetweetEngagement(test_dataset_id).save_feature(result)


class EngagerFeatureNumberOfPreviousCommentEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_comment_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.csv.gz")

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
        engagement_feature = TweetFeatureEngagementIsComment(train_dataset_id)

        # Save the column name
        eng_col = engagers_feature.feature_name

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        engager_counter_array = np.zeros(dataframe[engagers_feature.feature_name].max() + 1, dtype=int)

        result = pd.DataFrame(
            [find_and_increase(engager_id, engager_counter_array) if engagement else engager_counter_array[engager_id]
             for engager_id, engagement in zip(dataframe[eng_col], dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousCommentEngagement(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousCommentEngagement(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousCommentEngagement(test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
            ], axis=1)

            if dataframe[engagers_feature.feature_name].max() + 1 > engager_counter_array.size:
                engager_counter_array = np.pad(
                    engager_counter_array,
                    pad_width=(0, dataframe[engagers_feature.feature_name].max() + 1 - engager_counter_array.size),
                    mode='constant',
                    constant_values=0
                )

            result = pd.DataFrame(dataframe[eng_col].map(lambda x: engager_counter_array[x]),
                                  index=dataframe.index)
            EngagerFeatureNumberOfPreviousCommentEngagement(test_dataset_id).save_feature(result)


class EngagerFeatureNumberOfPreviousPositiveEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_positive_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.csv.gz")

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
        engagement_feature = TweetFeatureEngagementIsPositive(train_dataset_id)

        # Save the column name
        eng_col = engagers_feature.feature_name

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        engager_counter_array = np.zeros(dataframe[engagers_feature.feature_name].max() + 1, dtype=int)

        result = pd.DataFrame(
            [find_and_increase(engager_id, engager_counter_array) if engagement else engager_counter_array[engager_id]
             for engager_id, engagement in zip(dataframe[eng_col], dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousPositiveEngagement(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousPositiveEngagement(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousPositiveEngagement(test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
            ], axis=1)

            if dataframe[engagers_feature.feature_name].max() + 1 > engager_counter_array.size:
                engager_counter_array = np.pad(
                    engager_counter_array,
                    pad_width=(0, dataframe[engagers_feature.feature_name].max() + 1 - engager_counter_array.size),
                    mode='constant',
                    constant_values=0
                )

            result = pd.DataFrame(dataframe[eng_col].map(lambda x: engager_counter_array[x]),
                                  index=dataframe.index)
            EngagerFeatureNumberOfPreviousPositiveEngagement(test_dataset_id).save_feature(result)


class EngagerFeatureNumberOfPreviousNegativeEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_negative_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.csv.gz")

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
        engagement_feature = TweetFeatureEngagementIsNegative(train_dataset_id)

        # Save the column name
        eng_col = engagers_feature.feature_name

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
            engagement_feature.load_or_create()
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        engager_counter_array = np.zeros(dataframe[engagers_feature.feature_name].max() + 1, dtype=int)

        result = pd.DataFrame(
            [find_and_increase(engager_id, engager_counter_array) if engagement else engager_counter_array[engager_id]
             for engager_id, engagement in zip(dataframe[eng_col], dataframe[engagement_feature.feature_name])],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousNegativeEngagement(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousNegativeEngagement(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousNegativeEngagement(test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
            ], axis=1)

            if dataframe[engagers_feature.feature_name].max() + 1 > engager_counter_array.size:
                engager_counter_array = np.pad(
                    engager_counter_array,
                    pad_width=(0, dataframe[engagers_feature.feature_name].max() + 1 - engager_counter_array.size),
                    mode='constant',
                    constant_values=0
                )

            result = pd.DataFrame(dataframe[eng_col].map(lambda x: engager_counter_array[x]),
                                  index=dataframe.index)
            EngagerFeatureNumberOfPreviousNegativeEngagement(test_dataset_id).save_feature(result)


class EngagerFeatureNumberOfPreviousEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements/{self.feature_name}.csv.gz")

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

        # Save the column name
        eng_col = engagers_feature.feature_name

        dataframe = pd.concat([
            creation_timestamps_feature.load_or_create(),
            engagers_feature.load_or_create(),
        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        engager_counter_array = np.zeros(dataframe[engagers_feature.feature_name].max() + 1, dtype=int)

        result = pd.DataFrame(
            [find_and_increase(engager_id, engager_counter_array) for engager_id in dataframe[eng_col]],
            index=dataframe.index
        )

        if not EngagerFeatureNumberOfPreviousEngagement(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            EngagerFeatureNumberOfPreviousEngagement(train_dataset_id).save_feature(result)
        if not EngagerFeatureNumberOfPreviousEngagement(test_dataset_id).has_feature():
            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            engagers_feature = MappedFeatureEngagerId(test_dataset_id)

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                engagers_feature.load_or_create(),
            ], axis=1)

            if dataframe[engagers_feature.feature_name].max() + 1 > engager_counter_array.size:
                engager_counter_array = np.pad(
                    engager_counter_array,
                    pad_width=(0, dataframe[engagers_feature.feature_name].max() + 1 - engager_counter_array.size),
                    mode='constant',
                    constant_values=0
                )

            result = pd.DataFrame(dataframe[eng_col].map(lambda x: engager_counter_array[x]),
                                  index=dataframe.index)
            EngagerFeatureNumberOfPreviousEngagement(test_dataset_id).save_feature(result)
