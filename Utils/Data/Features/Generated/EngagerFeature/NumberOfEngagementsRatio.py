import numpy as np

from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId, MappedFeatureTweetHashtags
import time
from abc import ABC, abstractmethod


class NumberOfEngagementsRatioAbstract(GeneratedFeaturePickle, ABC):

    def __init__(self, dataset_id: str):
        super().__init__("number_of_engagements_ratio" + self._get_suffix(), dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_engagements_ratio/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_engagements_ratio/{self.feature_name}.csv.gz")
        self.number_of_folds = 10

    @abstractmethod
    def _get_suffix(self) -> str:
        pass

    @abstractmethod
    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        pass

    @classmethod
    def _save_train_result_if_not_present(cls, result, train_dataset_id):
        if not cls(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            cls(train_dataset_id).save_feature(result)

    @classmethod
    def _exists_test_feature(cls, test_dataset_id):
        return cls(test_dataset_id).has_feature()

    @classmethod
    def _save_test_result(cls, result, test_dataset_id):
        cls(test_dataset_id).save_feature(result)

    def create_feature(self):
        import Utils.Data.Data as data
        df = data.get_dataset([
            f"number_of_engagements_{self._get_suffix()}"
        ], self.dataset_id)
        support_df = data.get_dataset([
            f"number_of_engagements_positive",
            f"number_of_engagements_negative"
        ], self.dataset_id)

        df['total'] = support_df["number_of_engagements_positive"] + support_df["number_of_engagements_negative"]
        result = pd.DataFrame(df[f"number_of_engagements_{self._get_suffix()}"] / df["total"])
        result.fillna(0, inplace=True)
        result.replace([np.inf, -np.inf], 0, inplace=True)
        self.save_feature(result)


class NumberOfEngagementsRatioPositive(NumberOfEngagementsRatioAbstract):

    def _get_suffix(self) -> str:
        return "positive"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsPositive(dataset_id=dataset_id)


class NumberOfEngagementsRatioNegative(NumberOfEngagementsRatioAbstract):

    def _get_suffix(self) -> str:
        return "negative"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsNegative(dataset_id=dataset_id)


class NumberOfEngagementsRatioLike(NumberOfEngagementsRatioAbstract):

    def _get_suffix(self) -> str:
        return "like"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsLike(dataset_id=dataset_id)


class NumberOfEngagementsRatioRetweet(NumberOfEngagementsRatioAbstract):

    def _get_suffix(self) -> str:
        return "retweet"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsRetweet(dataset_id=dataset_id)


class NumberOfEngagementsRatioReply(NumberOfEngagementsRatioAbstract):

    def _get_suffix(self) -> str:
        return "reply"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsReply(dataset_id=dataset_id)


class NumberOfEngagementsRatioComment(NumberOfEngagementsRatioAbstract):

    def _get_suffix(self) -> str:
        return "comment"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsComment(dataset_id=dataset_id)

