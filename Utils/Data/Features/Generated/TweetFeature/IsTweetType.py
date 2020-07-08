from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle


class TweetFeatureIsReply(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_is_reply", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/tweet_type/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/tweet_type/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the media column
        feature = RawFeatureTweetType(self.dataset_id)
        feature_df = feature.load_or_create()
        type_id = "Reply"
        # Count the number of photos
        tweet_type_df = pd.DataFrame(feature_df[feature.feature_name].map(
            lambda x: True if x == type_id else False))

        self.save_feature(tweet_type_df)


class TweetFeatureIsRetweet(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_is_retweet", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/tweet_type/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/tweet_type/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the media column
        feature = RawFeatureTweetType(self.dataset_id)
        feature_df = feature.load_or_create()
        type_id = "Retweet"
        # Count the number of photos
        tweet_type_df = pd.DataFrame(feature_df[feature.feature_name].map(
            lambda x: True if x == type_id else False))

        self.save_feature(tweet_type_df)


class TweetFeatureIsQuote(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_is_quote", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/tweet_type/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/tweet_type/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the media column
        feature = RawFeatureTweetType(self.dataset_id)
        feature_df = feature.load_or_create()
        type_id = "Quote"
        # Count the number of photos
        tweet_type_df = pd.DataFrame(feature_df[feature.feature_name].map(
            lambda x: True if x == type_id else False))

        self.save_feature(tweet_type_df)

class TweetFeatureIsTopLevel(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_is_top_level", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/tweet_type/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/tweet_type/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the media column
        feature = RawFeatureTweetType(self.dataset_id)
        feature_df = feature.load_or_create()
        type_id = "TopLevel"
        # Count the number of photos
        tweet_type_df = pd.DataFrame(feature_df[feature.feature_name].map(
            lambda x: True if x == type_id else False))

        self.save_feature(tweet_type_df)