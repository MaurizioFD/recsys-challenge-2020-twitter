from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
from Utils.Data.Features.MappedFeatures import MappedFeatureTweetMedia


class TweetFeatureNumberOfPhoto(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_number_of_photo", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_media/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_media/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the media column
        feature = MappedFeatureTweetMedia(self.dataset_id)
        feature_df = feature.load_or_create()
        # Load the dictionary for mapping purpose
        dictionary = MappingMediaDictionary().load_or_create()
        media_id = dictionary["Photo"]
        # Count the number of photos
        number_of_media_df = pd.DataFrame(feature_df[feature.feature_name].map(
            lambda x: sum([1 for media in x if media == media_id]) if x is not None else 0))

        self.save_feature(number_of_media_df)


class TweetFeatureNumberOfVideo(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_number_of_video", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_media/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_media/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the media column
        feature = MappedFeatureTweetMedia(self.dataset_id)
        feature_df = feature.load_or_create()
        # Load the dictionary for mapping purpose
        dictionary = MappingMediaDictionary().load_or_create()
        media_id = dictionary["Video"]
        # Count the number of photos
        number_of_media_df = pd.DataFrame(feature_df[feature.feature_name].map(
            lambda x: sum([1 for media in x if media == media_id]) if x is not None else 0))

        self.save_feature(number_of_media_df)


class TweetFeatureNumberOfGif(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_number_of_gif", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_media/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_media/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the media column
        feature = MappedFeatureTweetMedia(self.dataset_id)
        feature_df = feature.load_or_create()
        # Load the dictionary for mapping purpose
        dictionary = MappingMediaDictionary().load_or_create()
        media_id = dictionary["GIF"]
        # Count the number of photos
        number_of_media_df = pd.DataFrame(feature_df[feature.feature_name].map(
            lambda x: sum([1 for media in x if media == media_id]) if x is not None else 0))

        self.save_feature(number_of_media_df)


class TweetFeatureNumberOfMedia(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_number_of_media", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_media/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_media/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the media column
        feature = MappedFeatureTweetMedia(self.dataset_id)
        feature_df = feature.load_or_create()
        # Count the number of photos
        number_of_media_df = pd.DataFrame(
            feature_df[feature.feature_name].map(lambda x: len(x) if x is not None else 0))

        self.save_feature(number_of_media_df)
