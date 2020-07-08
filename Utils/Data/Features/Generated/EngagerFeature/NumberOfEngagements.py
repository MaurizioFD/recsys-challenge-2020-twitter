import numpy as np

from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId, MappedFeatureTweetHashtags
import time
from abc import ABC, abstractmethod

def compute(train, test):
    dictionary = pd.DataFrame({"count": train.groupby("mapped_feature_engager_id").size()}).to_dict()["count"]
    result = pd.DataFrame(test["mapped_feature_engager_id"].map(lambda x: dictionary.get(x, 0)), index=test.index)
    return result

class NumberOfEngagementsAbstract(GeneratedFeaturePickle, ABC):

    def __init__(self, dataset_id: str):

        super().__init__("number_of_engagements_" + self._get_suffix(), dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_engagements/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_engagements/{self.feature_name}.csv.gz")
        self.number_of_folds = 20

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
        # Check if the dataset id is train or test
        if is_test_or_val_set(self.dataset_id):
            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)
            test_dataset_id = self.dataset_id
        else:
            train_dataset_id = self.dataset_id
            test_dataset_id = get_test_or_val_set_id_from_train(train_dataset_id)

        import Utils.Data.Data as data
        train_df = data.get_dataset([
            f"mapped_feature_engager_id",
            f"tweet_feature_engagement_is_{self._get_suffix()}"
        ], train_dataset_id)
        if is_test_or_val_set(self.dataset_id):
            test_df = data.get_dataset([
                f"mapped_feature_engager_id"
            ], test_dataset_id)
            train_df = train_df[train_df[f"tweet_feature_engagement_is_{self._get_suffix()}"] == True]
            res = compute(train_df, test_df)
            res.sort_index(inplace=True)
            self._save_test_result(res, test_dataset_id)
        else:
            # Compute the folds
            X_train_folds = np.array_split(train_df.sample(frac=1), self.number_of_folds)

            result = None

            for i in range(self.number_of_folds):
                local_train = pd.concat([X_train_folds[x] for x in range(self.number_of_folds) if x is not i])
                local_train = local_train[local_train[f"tweet_feature_engagement_is_{self._get_suffix()}"] == True]
                local_test = X_train_folds[i]

                res = compute(local_train, local_test)

                if result is None:
                    result = res
                else:
                    result = pd.concat([result, res])

            self._save_train_result_if_not_present(result, train_dataset_id)



class NumberOfEngagementsPositive(NumberOfEngagementsAbstract):

    def _get_suffix(self) -> str:
        return "positive"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsPositive(dataset_id=dataset_id)



class NumberOfEngagementsNegative(NumberOfEngagementsAbstract):

    def _get_suffix(self) -> str:
        return "negative"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsNegative(dataset_id=dataset_id)


class NumberOfEngagementsLike(NumberOfEngagementsAbstract):

    def _get_suffix(self) -> str:
        return "like"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsLike(dataset_id=dataset_id)


class NumberOfEngagementsRetweet(NumberOfEngagementsAbstract):

    def _get_suffix(self) -> str:
        return "retweet"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsRetweet(dataset_id=dataset_id)


class NumberOfEngagementsReply(NumberOfEngagementsAbstract):

    def _get_suffix(self) -> str:
        return "reply"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsReply(dataset_id=dataset_id)


class NumberOfEngagementsComment(NumberOfEngagementsAbstract):

    def _get_suffix(self) -> str:
        return "comment"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsComment(dataset_id=dataset_id)

