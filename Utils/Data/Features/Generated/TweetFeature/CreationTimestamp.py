from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
import pandas as pd
from datetime import datetime as dt


class TweetFeatureCreationTimestampHour(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_creation_timestamp_hour", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        feature = RawFeatureTweetTimestamp(self.dataset_id)
        feature_df = feature.load_or_create()
        # Map the hour of the timestamp
        hour_df = pd.DataFrame(feature_df[feature.feature_name].map(lambda x: dt.fromtimestamp(x).hour))

        self.save_feature(hour_df)

class TweetFeatureCreationTimestampWeekDay(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_creation_timestamp_week_day", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        feature = RawFeatureTweetTimestamp(self.dataset_id)
        feature_df = feature.load_or_create()
        # Map the week day of the timestamp
        week_day_df = pd.DataFrame(feature_df[feature.feature_name].map(lambda x: dt.fromtimestamp(x).weekday()))

        self.save_feature(week_day_df)


class TweetFeatureCreationTimestampHour_Shifted(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_creation_timestamp_hour_shifted", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        feature = TweetFeatureCreationTimestampHour(self.dataset_id)
        feature_df = feature.load_or_create()
        # Count the number of photos
        hour_df = pd.DataFrame(feature_df[feature.feature_name].map(lambda x: (x+12)%24))

        self.save_feature(hour_df)

class TweetFeatureCreationTimestampDayPhase(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_creation_timestamp_day_phase", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        feature = TweetFeatureCreationTimestampHour(self.dataset_id)
        feature_df = feature.load_or_create()
        # Count the number of photos
        week_day_df = pd.DataFrame(feature_df[feature.feature_name].map(lambda x: dayphase_dict[x]))

        self.save_feature(week_day_df)

class TweetFeatureCreationTimestampDayPhase_Shifted(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_creation_timestamp_day_phase_shifted", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creation_timestamp/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the engagement column
        feature = TweetFeatureCreationTimestampHour(self.dataset_id)
        feature_df = feature.load_or_create()
        # Count the number of photos
        week_day_df = pd.DataFrame(feature_df[feature.feature_name].map(lambda x: dayphase_dict[(x+12)%24]))

        self.save_feature(week_day_df)


#1 -> NIGHT
#2 -> NIGHT
#3 -> NIGHT
#4 -> NIGHT
#5 -> NIGHT
#6 -> MORNING
#7 -> MORNING
#8 -> MORNING
#9 -> MORNING
#10 -> MORINING
#11 -> LUNCH
#12 -> LUNCH
#13 -> LUNCH
#14 -> AFTERNOON
#15 -> AFTERNOON
#16 -> AFTERNOON
#17 -> AFTERNOON
#18 -> AFTERNOON
#19 -> EVENING
#20 -> EVENING
#21 -> EVENING
#22 -> NIGHT
#23 -> NIGHT

# NIGHT -> 0
# MORINING -> 1
# LUNCH -> 2
# AFTERNOON -> 3
# EVENING -> 4 


dayphase_dict = {
    0 : 0,
    1 : 0,
    2 : 0,
    3 : 0,
    4 : 0,
    5 : 0,
    6 : 1,
    7 : 1,
    8 : 1,
    9 : 1,
    10: 1,
    11: 2,
    12: 2,
    13: 2,
    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 3,
    19: 4,
    20: 4,
    21: 4,
    22: 0,
    23: 0
}