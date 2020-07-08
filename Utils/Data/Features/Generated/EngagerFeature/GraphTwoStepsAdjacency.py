import numpy as np

from Utils.Data.Features.Generated.EngagerFeature.GraphTwoSteps import *
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *

from abc import ABC, abstractmethod
from tqdm import tqdm


class GraphTwoStepsAdjacencyAbstract(GeneratedFeaturePickle, ABC):

    def __init__(self, dataset_id: str):

        super().__init__("graph_two_steps_adjacency_" + self._get_suffix(), dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/graph_two_steps/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/graph_two_steps/{self.feature_name}.csv.gz")
        self.number_of_folds = 20

    @abstractmethod
    def _get_suffix(self) -> str:
        pass

    @abstractmethod
    def _get_graph_two_steps_feature(self, dataset_id) -> GeneratedFeaturePickle:
        pass

    @classmethod
    def _exists_test_feature(cls, test_dataset_id):
        return cls(test_dataset_id).has_feature()

    @classmethod
    def _save_result(cls, result, dataset_id):
        cls(dataset_id).save_feature(result)

    def create_feature(self):
        graph_feat = self._get_graph_two_steps_feature(self.dataset_id)
        graph_feat_df = graph_feat.load_or_create()
        graph_feat_col = graph_feat.feature_name
        mask = (graph_feat_df[graph_feat_col] != 0)
        res = pd.DataFrame(mask)
        res.columns = [self.feature_name]
        self._save_result(result=res, dataset_id=self.dataset_id)


class GraphTwoStepsAdjacencyPositive(GraphTwoStepsAdjacencyAbstract):

    def _get_graph_two_steps_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return GraphTwoStepsPositive(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "positive"


class GraphTwoStepsAdjacencyNegative(GraphTwoStepsAdjacencyAbstract):

    def _get_graph_two_steps_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return GraphTwoStepsNegative(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "negative"


class GraphTwoStepsAdjacencyLike(GraphTwoStepsAdjacencyAbstract):

    def _get_graph_two_steps_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return GraphTwoStepsLike(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "like"


class GraphTwoStepsAdjacencyRetweet(GraphTwoStepsAdjacencyAbstract):

    def _get_graph_two_steps_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return GraphTwoStepsRetweet(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "retweet"


class GraphTwoStepsAdjacencyReply(GraphTwoStepsAdjacencyAbstract):

    def _get_graph_two_steps_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return GraphTwoStepsReply(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "reply"


class GraphTwoStepsAdjacencyComment(GraphTwoStepsAdjacencyAbstract):

    def _get_graph_two_steps_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return GraphTwoStepsComment(dataset_id=dataset_id)

    def _get_suffix(self) -> str:
        return "comment"
