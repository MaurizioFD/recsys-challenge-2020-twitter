from Utils.Data.Features.Generated.EngagerFeature.NumberOfEngagementsBetweenCreatorAndEngager import *
from abc import ABC, abstractmethod


class AdjacencyBetweenCreatorAndEngagerAbstract(GeneratedFeaturePickle, ABC):

    def __init__(self, dataset_id: str):

        super().__init__("adjacency_between_creator_and_engager_" + self._get_suffix(), dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/graph_two_steps/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/graph_two_steps/{self.feature_name}.csv.gz")
        self.number_of_folds = 20

    @abstractmethod
    def _get_suffix(self) -> str:
        pass

    @abstractmethod
    def _get_engagements_between_creator_and_engager_feature(self, dataset_id) -> GeneratedFeaturePickle:
        pass

    @classmethod
    def _exists_test_feature(cls, test_dataset_id):
        return cls(test_dataset_id).has_feature()

    @classmethod
    def _save_result(cls, result, dataset_id):
        cls(dataset_id).save_feature(result)

    def create_feature(self):
        eng_between_creator_engager_feat = \
            self._get_engagements_between_creator_and_engager_feature(self.dataset_id)
        eng_between_creator_engager_df = eng_between_creator_engager_feat.load_or_create()
        eng_between_creator_engager_feat_col = eng_between_creator_engager_feat.feature_name
        mask = (eng_between_creator_engager_df[eng_between_creator_engager_feat_col] != 0)
        res = pd.DataFrame(mask)
        res.columns = [self.feature_name]
        self._save_result(result=res, dataset_id=self.dataset_id)


class AdjacencyBetweenCreatorAndEngagerPositive(AdjacencyBetweenCreatorAndEngagerAbstract):

    def _get_engagements_between_creator_and_engager_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return NumberOfEngagementsBetweenCreatorAndEngagerPositive(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "positive"


class AdjacencyBetweenCreatorAndEngagerNegative(AdjacencyBetweenCreatorAndEngagerAbstract):

    def _get_engagements_between_creator_and_engager_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return NumberOfEngagementsBetweenCreatorAndEngagerNegative(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "negative"


class AdjacencyBetweenCreatorAndEngagerLike(AdjacencyBetweenCreatorAndEngagerAbstract):

    def _get_engagements_between_creator_and_engager_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return NumberOfEngagementsBetweenCreatorAndEngagerLike(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "like"


class AdjacencyBetweenCreatorAndEngagerRetweet(AdjacencyBetweenCreatorAndEngagerAbstract):

    def _get_engagements_between_creator_and_engager_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return NumberOfEngagementsBetweenCreatorAndEngagerRetweet(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "retweet"


class AdjacencyBetweenCreatorAndEngagerReply(AdjacencyBetweenCreatorAndEngagerAbstract):

    def _get_engagements_between_creator_and_engager_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return NumberOfEngagementsBetweenCreatorAndEngagerReply(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "reply"


class AdjacencyBetweenCreatorAndEngagerComment(AdjacencyBetweenCreatorAndEngagerAbstract):

    def _get_engagements_between_creator_and_engager_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return NumberOfEngagementsBetweenCreatorAndEngagerComment(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "comment"
