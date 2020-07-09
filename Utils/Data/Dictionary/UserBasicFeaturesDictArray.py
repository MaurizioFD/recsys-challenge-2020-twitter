import pandas as pd
import pathlib as pl
import numpy as np
import RootPath
from abc import abstractmethod

from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import TweetFeatureEngagementIsPositive
from Utils.Data.Features.MappedFeatures import *
from Utils.Data.Dictionary.MappingDictionary import *


class UserBasicFeatureDictArrayNumpy(Dictionary):
    """
    It is built only using train and test set.
    Abstract class representing a dictionary array that works with numpy/pickle file.
    """

    def __init__(self, dictionary_name: str):
        super().__init__(dictionary_name)
        self.npz_path = pl.Path(f"{Dictionary.ROOT_PATH}/basic_features/user/{self.dictionary_name}.npz")

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


class FollowerCountUserBasicFeatureDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("follower_count_user_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_engager_id_feature = MappedFeatureEngagerId("train")
        test_engager_id_feature = MappedFeatureEngagerId("test")
        last_test_engager_id_feature = MappedFeatureEngagerId("last_test")
        train_creator_id_feature = MappedFeatureCreatorId("train")
        test_creator_id_feature = MappedFeatureCreatorId("test")
        last_test_creator_id_feature = MappedFeatureCreatorId("last_test")

        # Find the mask of uniques one
        engager_train_df = train_engager_id_feature.load_or_create()
        engager_test_df = test_engager_id_feature.load_or_create()
        engager_last_test_df = last_test_engager_id_feature.load_or_create()
        engager_test_df = pd.concat([engager_test_df, engager_last_test_df])
        creator_train_df = train_creator_id_feature.load_or_create()
        creator_test_df = test_creator_id_feature.load_or_create()
        creator_last_test_df = last_test_creator_id_feature.load_or_create()
        creator_test_df = pd.concat([creator_test_df, creator_last_test_df])

        unique = ~engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name])[unique]

        # Load the target column
        column = "follower_count"
        engager_train_target_feature = RawFeatureEngagerFollowerCount("train")
        engager_test_target_feature = RawFeatureEngagerFollowerCount("test")
        creator_train_target_feature = RawFeatureCreatorFollowerCount("train")
        creator_test_target_feature = RawFeatureCreatorFollowerCount("test")
        engager_last_test_target_feature = RawFeatureEngagerFollowerCount("last_test")
        creator_last_test_target_feature = RawFeatureCreatorFollowerCount("last_test")
        engager_train_df = engager_train_target_feature.load_or_create()
        engager_test_df = engager_test_target_feature.load_or_create()
        creator_train_df = creator_train_target_feature.load_or_create()
        creator_test_df = creator_test_target_feature.load_or_create()
        last_engager_test_df = engager_last_test_target_feature.load_or_create()
        last_creator_test_df = creator_last_test_target_feature.load_or_create()
        engager_test_df = pd.concat([engager_test_df, last_engager_test_df])
        creator_test_df = pd.concat([creator_test_df, last_creator_test_df])
        df[column] = engager_train_df[engager_train_target_feature.feature_name].append(
            engager_test_df[engager_test_target_feature.feature_name]).append(
            creator_train_df[creator_train_target_feature.feature_name]).append(
            creator_test_df[creator_test_target_feature.feature_name])[unique]

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class FollowingCountUserBasicFeatureDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("following_count_user_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_engager_id_feature = MappedFeatureEngagerId("train")
        test_engager_id_feature = MappedFeatureEngagerId("test")
        last_test_engager_id_feature = MappedFeatureEngagerId("last_test")
        train_creator_id_feature = MappedFeatureCreatorId("train")
        test_creator_id_feature = MappedFeatureCreatorId("test")
        last_test_creator_id_feature = MappedFeatureCreatorId("last_test")

        # Find the mask of uniques one
        engager_train_df = train_engager_id_feature.load_or_create()
        engager_test_df = test_engager_id_feature.load_or_create()
        engager_last_test_df = last_test_engager_id_feature.load_or_create()
        engager_test_df = pd.concat([engager_test_df, engager_last_test_df])
        creator_train_df = train_creator_id_feature.load_or_create()
        creator_test_df = test_creator_id_feature.load_or_create()
        creator_last_test_df = last_test_creator_id_feature.load_or_create()
        creator_test_df = pd.concat([creator_test_df, creator_last_test_df])

        unique = ~engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name])[unique]

        # Load the target column
        column = "following_count"
        engager_train_target_feature = RawFeatureEngagerFollowingCount("train")
        engager_test_target_feature = RawFeatureEngagerFollowingCount("test")
        creator_train_target_feature = RawFeatureCreatorFollowingCount("train")
        creator_test_target_feature = RawFeatureCreatorFollowingCount("test")
        engager_last_test_target_feature = RawFeatureCreatorFollowingCount("last_test")
        creator_last_test_target_feature = RawFeatureCreatorFollowingCount("last_test")
        engager_train_df = engager_train_target_feature.load_or_create()
        engager_test_df = engager_test_target_feature.load_or_create()
        creator_train_df = creator_train_target_feature.load_or_create()
        creator_test_df = creator_test_target_feature.load_or_create()
        last_engager_test_df = engager_last_test_target_feature.load_or_create()
        last_creator_test_df = creator_last_test_target_feature.load_or_create()
        engager_test_df = pd.concat([engager_test_df, last_engager_test_df])
        creator_test_df = pd.concat([creator_test_df, last_creator_test_df])
        df[column] = engager_train_df[engager_train_target_feature.feature_name].append(
            engager_test_df[engager_test_target_feature.feature_name]).append(
            creator_train_df[creator_train_target_feature.feature_name]).append(
            creator_test_df[creator_test_target_feature.feature_name])[unique]

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class IsVerifiedUserBasicFeatureDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("is_verified_user_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_engager_id_feature = MappedFeatureEngagerId("train")
        test_engager_id_feature = MappedFeatureEngagerId("test")
        last_test_engager_id_feature = MappedFeatureEngagerId("last_test")
        train_creator_id_feature = MappedFeatureCreatorId("train")
        test_creator_id_feature = MappedFeatureCreatorId("test")
        last_test_creator_id_feature = MappedFeatureCreatorId("last_test")

        # Find the mask of uniques one
        engager_train_df = train_engager_id_feature.load_or_create()
        engager_test_df = test_engager_id_feature.load_or_create()
        engager_last_test_df = last_test_engager_id_feature.load_or_create()
        engager_test_df = pd.concat([engager_test_df, engager_last_test_df])
        creator_train_df = train_creator_id_feature.load_or_create()
        creator_test_df = test_creator_id_feature.load_or_create()
        creator_last_test_df = last_test_creator_id_feature.load_or_create()
        creator_test_df = pd.concat([creator_test_df, creator_last_test_df])

        unique = ~engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name])[unique]

        # Load the target column
        column = "is_verified"
        engager_train_target_feature = RawFeatureEngagerIsVerified("train")
        engager_test_target_feature = RawFeatureEngagerIsVerified("test")
        creator_train_target_feature = RawFeatureCreatorIsVerified("train")
        creator_test_target_feature = RawFeatureCreatorIsVerified("test")
        engager_last_test_target_feature = RawFeatureCreatorIsVerified("last_test")
        creator_last_test_target_feature = RawFeatureCreatorIsVerified("last_test")
        engager_train_df = engager_train_target_feature.load_or_create()
        engager_test_df = engager_test_target_feature.load_or_create()
        creator_train_df = creator_train_target_feature.load_or_create()
        creator_test_df = creator_test_target_feature.load_or_create()
        last_engager_test_df = engager_last_test_target_feature.load_or_create()
        last_creator_test_df = creator_last_test_target_feature.load_or_create()
        engager_test_df = pd.concat([engager_test_df, last_engager_test_df])
        creator_test_df = pd.concat([creator_test_df, last_creator_test_df])
        df[column] = engager_train_df[engager_train_target_feature.feature_name].append(
            engager_test_df[engager_test_target_feature.feature_name]).append(
            creator_train_df[creator_train_target_feature.feature_name]).append(
            creator_test_df[creator_test_target_feature.feature_name])[unique]

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)


class CreationTimestampUserBasicFeatureDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("creation_timestamp_user_dict_array")

    def create_dictionary(self):
        df = pd.DataFrame()

        # Load the index column
        train_engager_id_feature = MappedFeatureEngagerId("train")
        test_engager_id_feature = MappedFeatureEngagerId("test")
        last_test_engager_id_feature = MappedFeatureEngagerId("last_test")
        train_creator_id_feature = MappedFeatureCreatorId("train")
        test_creator_id_feature = MappedFeatureCreatorId("test")
        last_test_creator_id_feature = MappedFeatureCreatorId("last_test")

        # Find the mask of uniques one
        engager_train_df = train_engager_id_feature.load_or_create()
        engager_test_df = test_engager_id_feature.load_or_create()
        engager_last_test_df = last_test_engager_id_feature.load_or_create()
        engager_test_df = pd.concat([engager_test_df, engager_last_test_df])
        creator_train_df = train_creator_id_feature.load_or_create()
        creator_test_df = test_creator_id_feature.load_or_create()
        creator_last_test_df = last_test_creator_id_feature.load_or_create()
        creator_test_df = pd.concat([creator_test_df, creator_last_test_df])

        unique = ~engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name]).duplicated()

        # Unique tweet ids column
        df['id'] = engager_train_df[train_engager_id_feature.feature_name].append(
            engager_test_df[test_engager_id_feature.feature_name]).append(
            creator_train_df[train_creator_id_feature.feature_name]).append(
            creator_test_df[test_creator_id_feature.feature_name])[unique]

        # Load the target column
        column = "creation_timestamp"
        engager_train_target_feature = RawFeatureEngagerCreationTimestamp("train")
        engager_test_target_feature = RawFeatureEngagerCreationTimestamp("test")
        creator_train_target_feature = RawFeatureCreatorCreationTimestamp("train")
        creator_test_target_feature = RawFeatureCreatorCreationTimestamp("test")
        engager_last_test_target_feature = RawFeatureCreatorCreationTimestamp("last_test")
        creator_last_test_target_feature = RawFeatureCreatorCreationTimestamp("last_test")
        engager_train_df = engager_train_target_feature.load_or_create()
        engager_test_df = engager_test_target_feature.load_or_create()
        creator_train_df = creator_train_target_feature.load_or_create()
        creator_test_df = creator_test_target_feature.load_or_create()
        last_engager_test_df = engager_last_test_target_feature.load_or_create()
        last_creator_test_df = creator_last_test_target_feature.load_or_create()
        engager_test_df = pd.concat([engager_test_df, last_engager_test_df])
        creator_test_df = pd.concat([creator_test_df, last_creator_test_df])
        df[column] = engager_train_df[engager_train_target_feature.feature_name].append(
            engager_test_df[engager_test_target_feature.feature_name]).append(
            creator_train_df[creator_train_target_feature.feature_name]).append(
            creator_test_df[creator_test_target_feature.feature_name])[unique]

        # Cast it to a numpy array
        arr = np.array(df.sort_values(by='id')[column].array)

        self.save_dictionary(arr)

class LanguageUserBasicFeatureDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("language_user_dict_array")

    def create_dictionary(self):
        result = pd.DataFrame()
        train_dataset_id = "train"
        test_dataset_id = "test"

        # Load the index column
        train_engager_id_feature = MappedFeatureEngagerId(train_dataset_id)
        test_engager_id_feature = MappedFeatureEngagerId(test_dataset_id)
        train_creator_id_feature = MappedFeatureCreatorId(train_dataset_id)
        test_creator_id_feature = MappedFeatureCreatorId(test_dataset_id)
        train_language_id_feature = MappedFeatureTweetLanguage(train_dataset_id)
        test_language_id_feature = MappedFeatureTweetLanguage(test_dataset_id)
        train_is_positive_feature = TweetFeatureEngagementIsPositive(train_dataset_id)
        train_tweet_id_feature = MappedFeatureTweetId(train_dataset_id)
        test_tweet_id_feature = MappedFeatureTweetId(test_dataset_id)

        # Find the mask of uniques one
        engager_train_df = train_engager_id_feature.load_or_create()
        engager_test_df = test_engager_id_feature.load_or_create()
        creator_train_df = train_creator_id_feature.load_or_create()
        creator_test_df = test_creator_id_feature.load_or_create()
        is_positive_train_df = train_is_positive_feature.load_or_create()
        language_train_df = train_language_id_feature.load_or_create()
        language_test_df = test_language_id_feature.load_or_create()
        tweet_id_train_df = train_tweet_id_feature.load_or_create()
        tweet_id_test_df = test_tweet_id_feature.load_or_create()

        # Set index
        result['user'] = range(max(
            engager_train_df[train_engager_id_feature.feature_name].max(),
            engager_test_df[test_engager_id_feature.feature_name].max(),
            creator_train_df[train_creator_id_feature.feature_name].max(),
            creator_test_df[test_creator_id_feature.feature_name].max()
        ) + 1)
        result.set_index('user', inplace=True)

        # Create the creator dataframe
        creator_df = pd.concat(
            [
                creator_test_df.append(creator_train_df),
                language_test_df.append(language_train_df),
                tweet_id_test_df.append(tweet_id_train_df)
            ], axis=1
        ).drop_duplicates(train_tweet_id_feature.feature_name).drop(columns=train_tweet_id_feature.feature_name)

        creator_df.columns = ["user", "language"]

        # Create the engager dataframe
        engager_df = pd.concat(
            [
                engager_train_df[is_positive_train_df[train_is_positive_feature.feature_name]],
                language_train_df[is_positive_train_df[train_is_positive_feature.feature_name]]
            ], axis=1
        )

        engager_df.columns = ["user", "language"]

        dataframe = pd.concat(
            [
                creator_df,
                engager_df
            ]
        )

        # Group by and aggregate in numpy array
        result['language'] = dataframe.groupby("user").agg(list)['language'].apply(
            lambda x: np.array(x, dtype=np.uint8))
        result['language'].replace({np.nan: None}, inplace=True)

        # To numpy array
        arr = np.array(result['language'].array)

        self.save_dictionary(arr)
