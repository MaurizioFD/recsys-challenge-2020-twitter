from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
import pandas as pd


class TweetFeatureEngagementIsLike(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_engagement_is_like", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        feature = RawFeatureEngagementLikeTimestamp(self.dataset_id)
        feature_df = feature.load_or_create()
        # Count the number of photos
        tweet_engagement_type_df = pd.DataFrame(feature_df[feature.feature_name].map(
            lambda x: True if x is not pd.NA else False))

        self.save_feature(tweet_engagement_type_df)


class TweetFeatureEngagementIsRetweet(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_engagement_is_retweet", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        feature = RawFeatureEngagementRetweetTimestamp(self.dataset_id)
        feature_df = feature.load_or_create()
        # Count the number of photos
        tweet_engagement_type_df = pd.DataFrame(feature_df[feature.feature_name].map(
            lambda x: True if x is not pd.NA else False))

        self.save_feature(tweet_engagement_type_df)


class TweetFeatureEngagementIsComment(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_engagement_is_comment", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        feature = RawFeatureEngagementCommentTimestamp(self.dataset_id)
        feature_df = feature.load_or_create()
        # Count the number of photos
        tweet_engagement_type_df = pd.DataFrame(feature_df[feature.feature_name].map(
            lambda x: True if x is not pd.NA else False))

        self.save_feature(tweet_engagement_type_df)


class TweetFeatureEngagementIsReply(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_engagement_is_reply", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        feature = RawFeatureEngagementReplyTimestamp(self.dataset_id)
        feature_df = feature.load_or_create()
        # Count the number of photos
        tweet_engagement_type_df = pd.DataFrame(feature_df[feature.feature_name].map(
            lambda x: True if x is not pd.NA else False))

        self.save_feature(tweet_engagement_type_df)


class TweetFeatureEngagementIsPositive(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_engagement_is_positive", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        is_reply_feature = TweetFeatureEngagementIsReply(self.dataset_id).load_or_create()
        is_comment_feature = TweetFeatureEngagementIsComment(self.dataset_id).load_or_create()
        is_like_feature = TweetFeatureEngagementIsLike(self.dataset_id).load_or_create()
        is_retweet_feature = TweetFeatureEngagementIsRetweet(self.dataset_id).load_or_create()
        df = pd.concat(
            [
                is_reply_feature,
                is_comment_feature,
                is_like_feature,
                is_retweet_feature
            ],
            axis=1
        )
        is_positive_df = pd.DataFrame(df.any(axis=1))
        self.save_feature(is_positive_df)

class TweetFeatureEngagementIsNegative(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_engagement_is_negative", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/engagement_type/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        is_positive_df = TweetFeatureEngagementIsPositive(self.dataset_id).load_or_create()
        is_negative_df = ~is_positive_df

        self.save_feature(is_negative_df)
