from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
from Utils.Data.Features.Generated.EngagerFeature.NumberOfPreviousEngagements import *


class EngagerFeatureNumberOfPreviousLikeEngagementRatio(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_like_engagement_ratio", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousLikeEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(0, inplace=True)
        result.replace([np.inf, -np.inf], 0, inplace=True)

        self.save_feature(result)


class EngagerFeatureNumberOfPreviousReplyEngagementRatio(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_reply_engagement_ratio", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousReplyEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(0, inplace=True)
        result.replace([np.inf, -np.inf], 0, inplace=True)

        self.save_feature(result)


class EngagerFeatureNumberOfPreviousRetweetEngagementRatio(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_retweet_engagement_ratio", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousRetweetEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(0, inplace=True)
        result.replace([np.inf, -np.inf], 0, inplace=True)

        self.save_feature(result)


class EngagerFeatureNumberOfPreviousCommentEngagementRatio(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_comment_engagement_ratio", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousCommentEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(0, inplace=True)
        result.replace([np.inf, -np.inf], 0, inplace=True)

        self.save_feature(result)


class EngagerFeatureNumberOfPreviousPositiveEngagementRatio(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_positive_engagement_ratio", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousPositiveEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(0, inplace=True)
        result.replace([np.inf, -np.inf], 0, inplace=True)

        self.save_feature(result)


class EngagerFeatureNumberOfPreviousNegativeEngagementRatio(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_negative_engagement_ratio", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousNegativeEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(0, inplace=True)
        result.replace([np.inf, -np.inf], 0, inplace=True)

        self.save_feature(result)

class EngagerFeatureNumberOfPreviousLikeEngagementRatio1(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_like_engagement_ratio_1", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousLikeEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(-1, inplace=True)
        result.replace([np.inf, -np.inf], -1, inplace=True)

        self.save_feature(result)


class EngagerFeatureNumberOfPreviousReplyEngagementRatio1(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_reply_engagement_ratio_1", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousReplyEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(-1, inplace=True)
        result.replace([np.inf, -np.inf], -1, inplace=True)

        self.save_feature(result)


class EngagerFeatureNumberOfPreviousRetweetEngagementRatio1(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_retweet_engagement_ratio_1", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousRetweetEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(-1, inplace=True)
        result.replace([np.inf, -np.inf], -1, inplace=True)

        self.save_feature(result)


class EngagerFeatureNumberOfPreviousCommentEngagementRatio1(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_comment_engagement_ratio_1", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousCommentEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(-1, inplace=True)
        result.replace([np.inf, -np.inf], -1, inplace=True)

        self.save_feature(result)


class EngagerFeatureNumberOfPreviousPositiveEngagementRatio1(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_positive_engagement_ratio_1", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousPositiveEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(-1, inplace=True)
        result.replace([np.inf, -np.inf], -1, inplace=True)

        self.save_feature(result)


class EngagerFeatureNumberOfPreviousNegativeEngagementRatio1(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_number_of_previous_negative_engagement_ratio_1", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_previous_engagements_ratio/{self.feature_name}.csv.gz")

    def create_feature(self):
        engagement_feature = EngagerFeatureNumberOfPreviousNegativeEngagement(self.dataset_id)
        overall_feature = EngagerFeatureNumberOfPreviousEngagement(self.dataset_id)

        result = pd.DataFrame(
            engagement_feature.load_or_create()[engagement_feature.feature_name] / overall_feature.load_or_create()[
                overall_feature.feature_name]
        )

        result.fillna(-1, inplace=True)
        result.replace([np.inf, -np.inf], -1, inplace=True)

        self.save_feature(result)