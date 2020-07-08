from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
from Utils.Data.Features.MappedFeatures import MappedFeatureTweetHashtags


class TweetFeatureNumberOfHashtags(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_number_of_hashtags", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_hashtags/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_hashtags/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the hashtags column
        feature = MappedFeatureTweetHashtags(self.dataset_id)
        feature_df = feature.load_or_create()
        # Count the number of hashtags
        number_of_hashtags_df = pd.DataFrame(
            feature_df[feature.feature_name].map(lambda x: len(x) if x is not None else 0))

        self.save_feature(number_of_hashtags_df)