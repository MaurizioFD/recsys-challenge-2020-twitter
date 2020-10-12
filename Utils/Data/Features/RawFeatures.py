from Utils.Data.Features.Feature import Feature
import pandas as pd
import pathlib as pl
import RootPath


class RawFeatureCSV(Feature):
    """
    Abstract class representing a feature in raw format that works with csv file.
    It is needed in order to cope with NAN/NA values.
    """

    def __init__(self, feature_name: str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.csv_path = pl.Path(f"{Feature.ROOT_PATH}/{self.dataset_id}/raw/{self.feature_name}.csv.gz")

    def has_feature(self):
        return self.csv_path.is_file()

    def load_feature(self):
        assert self.has_feature(), f"The feature {self.feature_name} does not exists. Create it first."
        df = pd.read_csv(self.csv_path, compression="gzip", index_col=0, header=0,
                         dtype={self.feature_name: pd.StringDtype()})
        # Renaming the column for consistency purpose
        df.columns = [self.feature_name]
        return df

    def load_feature_reader(self, chunksize=1000):
        if not self.has_feature():
            self.create_feature()

        reader = pd.read_csv(self.csv_path, compression="gzip", index_col=0, header=0,
                         dtype={self.feature_name: pd.StringDtype()}, chunksize=chunksize)
        return reader

    def create_feature(self):
        df = _get_raw_column(self.feature_name, self.dataset_id)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.csv_path, compression='gzip', index=True)


class RawFeaturePickle(Feature):
    """
    Abstract class representing a feature in raw format that works with pickle file.
    """

    def __init__(self, feature_name: str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(f"{Feature.ROOT_PATH}/{self.dataset_id}/raw/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(f"{Feature.ROOT_PATH}/{self.dataset_id}/raw/{self.feature_name}.csv.gz")

    def has_feature(self):
        return self.pck_path.is_file()

    def load_feature(self):
        assert self.has_feature(), f"The feature {self.feature_name} does not exists. Create it first."
        df = pd.read_pickle(self.pck_path, compression="gzip")
        # Renaming the column for consistency purpose
        df.columns = [self.feature_name]
        return df

    def create_feature(self):
        df = _get_raw_column(self.feature_name, self.dataset_id)
        self.pck_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(self.pck_path, compression='gzip')
        # For backup reason
        # self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # df.to_csv(self.csv_path, compression='gzip', index=True)


class RawFeatureTweetTextToken(RawFeatureCSV):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_tweet_text_token", dataset_id)


class RawFeatureTweetHashtags(RawFeatureCSV):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_tweet_hashtags", dataset_id)


class RawFeatureTweetId(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_tweet_id", dataset_id)


class RawFeatureTweetMedia(RawFeatureCSV):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_tweet_media", dataset_id)


class RawFeatureTweetLinks(RawFeatureCSV):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_tweet_links", dataset_id)


class RawFeatureTweetDomains(RawFeatureCSV):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_tweet_domains", dataset_id)


class RawFeatureTweetType(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_tweet_type", dataset_id)


class RawFeatureTweetLanguage(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_tweet_language", dataset_id)


class RawFeatureTweetTimestamp(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_tweet_timestamp", dataset_id)


class RawFeatureCreatorId(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_creator_id", dataset_id)


class RawFeatureCreatorFollowerCount(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_creator_follower_count", dataset_id)


class RawFeatureCreatorFollowingCount(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_creator_following_count", dataset_id)


class RawFeatureCreatorIsVerified(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_creator_is_verified", dataset_id)


class RawFeatureCreatorCreationTimestamp(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_creator_creation_timestamp", dataset_id)


class RawFeatureEngagerId(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_engager_id", dataset_id)


class RawFeatureEngagerFollowerCount(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_engager_follower_count", dataset_id)


class RawFeatureEngagerFollowingCount(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_engager_following_count", dataset_id)


class RawFeatureEngagerIsVerified(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_engager_is_verified", dataset_id)


class RawFeatureEngagerCreationTimestamp(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_engager_creation_timestamp", dataset_id)


class RawFeatureEngagementCreatorFollowsEngager(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_engagement_creator_follows_engager", dataset_id)


class RawFeatureEngagementReplyTimestamp(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_engagement_reply_timestamp", dataset_id)


class RawFeatureEngagementRetweetTimestamp(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_engagement_retweet_timestamp", dataset_id)


class RawFeatureEngagementCommentTimestamp(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_engagement_comment_timestamp", dataset_id)


class RawFeatureEngagementLikeTimestamp(RawFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("raw_feature_engagement_like_timestamp", dataset_id)


def get_raw_column(column, dataset_id):
    df = pd.read_csv(f"{RootPath.get_dataset_path()}/{dataset_id}.csv.gz", compression='gzip', sep='', nrows=1)
    columns = len(df.columns)
    if columns == 20:
        return pd.read_csv(f"{RootPath.get_dataset_path()}/{dataset_id}.csv.gz",
                           compression='gzip',
                           sep='',
                           names=[
                               "raw_feature_tweet_text_token",
                               "raw_feature_tweet_hashtags",
                               "raw_feature_tweet_id",
                               "raw_feature_tweet_media",
                               "raw_feature_tweet_links",
                               "raw_feature_tweet_domains",
                               "raw_feature_tweet_type",
                               "raw_feature_tweet_language",
                               "raw_feature_tweet_timestamp",
                               "raw_feature_creator_id",
                               "raw_feature_creator_follower_count",
                               "raw_feature_creator_following_count",
                               "raw_feature_creator_is_verified",
                               "raw_feature_creator_creation_timestamp",
                               "raw_feature_engager_id",
                               "raw_feature_engager_follower_count",
                               "raw_feature_engager_following_count",
                               "raw_feature_engager_is_verified",
                               "raw_feature_engager_creation_timestamp",
                               "raw_feature_engagement_creator_follows_engager"
                           ],
                           dtype={
                               "raw_feature_tweet_text_token": pd.StringDtype(),
                               "raw_feature_tweet_hashtags": pd.StringDtype(),
                               "raw_feature_tweet_id": pd.StringDtype(),
                               "raw_feature_tweet_media": pd.StringDtype(),
                               "raw_feature_tweet_links": pd.StringDtype(),
                               "raw_feature_tweet_domains": pd.StringDtype(),
                               "raw_feature_tweet_type": pd.StringDtype(),
                               "raw_feature_tweet_language": pd.StringDtype(),
                               "raw_feature_tweet_timestamp": pd.Int32Dtype(),
                               "raw_feature_creator_id": pd.StringDtype(),
                               "raw_feature_creator_follower_count": pd.Int32Dtype(),
                               "raw_feature_creator_following_count": pd.Int32Dtype(),
                               "raw_feature_creator_is_verified": pd.BooleanDtype(),
                               "raw_feature_creator_creation_timestamp": pd.Int32Dtype(),
                               "raw_feature_engager_id": pd.StringDtype(),
                               "raw_feature_engager_follower_count": pd.Int32Dtype(),
                               "raw_feature_engager_following_count": pd.Int32Dtype(),
                               "raw_feature_engager_is_verified": pd.BooleanDtype(),
                               "raw_feature_engager_creation_timestamp": pd.Int32Dtype(),
                               "raw_feature_engagement_creator_follows_engager": pd.BooleanDtype()
                           },
                           usecols=[
                               column
                           ]
                           )
    # Read the dataframe
    elif columns == 24:
        return pd.read_csv(f"{RootPath.get_dataset_path()}/{dataset_id}.csv.gz",
                           compression='gzip',
                           sep='',
                           names=[
                               "raw_feature_tweet_text_token",
                               "raw_feature_tweet_hashtags",
                               "raw_feature_tweet_id",
                               "raw_feature_tweet_media",
                               "raw_feature_tweet_links",
                               "raw_feature_tweet_domains",
                               "raw_feature_tweet_type",
                               "raw_feature_tweet_language",
                               "raw_feature_tweet_timestamp",
                               "raw_feature_creator_id",
                               "raw_feature_creator_follower_count",
                               "raw_feature_creator_following_count",
                               "raw_feature_creator_is_verified",
                               "raw_feature_creator_creation_timestamp",
                               "raw_feature_engager_id",
                               "raw_feature_engager_follower_count",
                               "raw_feature_engager_following_count",
                               "raw_feature_engager_is_verified",
                               "raw_feature_engager_creation_timestamp",
                               "raw_feature_engagement_creator_follows_engager",
                               "raw_feature_engagement_reply_timestamp",
                               "raw_feature_engagement_retweet_timestamp",
                               "raw_feature_engagement_comment_timestamp",
                               "raw_feature_engagement_like_timestamp"
                           ],
                           dtype={
                               "raw_feature_tweet_text_token": pd.StringDtype(),
                               "raw_feature_tweet_hashtags": pd.StringDtype(),
                               "raw_feature_tweet_id": pd.StringDtype(),
                               "raw_feature_tweet_media": pd.StringDtype(),
                               "raw_feature_tweet_links": pd.StringDtype(),
                               "raw_feature_tweet_domains": pd.StringDtype(),
                               "raw_feature_tweet_type": pd.StringDtype(),
                               "raw_feature_tweet_language": pd.StringDtype(),
                               "raw_feature_tweet_timestamp": pd.Int32Dtype(),
                               "raw_feature_creator_id": pd.StringDtype(),
                               "raw_feature_creator_follower_count": pd.Int32Dtype(),
                               "raw_feature_creator_following_count": pd.Int32Dtype(),
                               "raw_feature_creator_is_verified": pd.BooleanDtype(),
                               "raw_feature_creator_creation_timestamp": pd.Int32Dtype(),
                               "raw_feature_engager_id": pd.StringDtype(),
                               "raw_feature_engager_follower_count": pd.Int32Dtype(),
                               "raw_feature_engager_following_count": pd.Int32Dtype(),
                               "raw_feature_engager_is_verified": pd.BooleanDtype(),
                               "raw_feature_engager_creation_timestamp": pd.Int32Dtype(),
                               "raw_feature_engagement_creator_follows_engager": pd.BooleanDtype(),
                               "raw_feature_engagement_reply_timestamp": pd.Int32Dtype(),
                               "raw_feature_engagement_retweet_timestamp": pd.Int32Dtype(),
                               "raw_feature_engagement_comment_timestamp": pd.Int32Dtype(),
                               "raw_feature_engagement_like_timestamp": pd.Int32Dtype()
                           },
                           usecols=[
                               column
                           ]
                           )
    else:
        raise Exception("something went wrong.")
