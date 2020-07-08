import numpy as np

from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId, MappedFeatureTweetHashtags
import time
from abc import ABC, abstractmethod


def compute(train, test, label):
    creator_dictionary = pd.DataFrame(
        {"count": train[[
            f"mapped_feature_creator_id",
            f"mapped_feature_tweet_language"
        ]].groupby([f"mapped_feature_creator_id", f"mapped_feature_tweet_language"]).size()}
    ).to_dict()["count"]
    engager_dictionary = pd.DataFrame(
        {"count": train[train[label] == True][[
            f"mapped_feature_engager_id",
            f"mapped_feature_tweet_language"
        ]].groupby([f"mapped_feature_engager_id", f"mapped_feature_tweet_language"]).size()}
    ).to_dict()["count"]
    dictionary = {**creator_dictionary, **engager_dictionary}
    result = pd.DataFrame(
        [
            dictionary.get((engager, language), 0)
            for engager, language
            in zip(test[f"mapped_feature_engager_id"], test[f"mapped_feature_tweet_language"])
        ],
        index=test.index)
    return result


class NumberOfEngagementsWithLanguageAbstract(GeneratedFeaturePickle, ABC):

    def __init__(self, dataset_id: str):

        super().__init__("number_of_engagements_with_language_" + self._get_suffix(), dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_engagements_with_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/number_of_engagements_with_language/{self.feature_name}.csv.gz")
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
            f"mapped_feature_creator_id",
            f"mapped_feature_engager_id",
            f"mapped_feature_tweet_language",
            f"tweet_feature_engagement_is_{self._get_suffix()}"
        ], train_dataset_id)
        if is_test_or_val_set(self.dataset_id):
            test_df = data.get_dataset([
                f"mapped_feature_creator_id",
                f"mapped_feature_engager_id",
                f"mapped_feature_tweet_language"
            ], test_dataset_id)
            res = compute(train_df, test_df, f"tweet_feature_engagement_is_{self._get_suffix()}")
            res.sort_index(inplace=True)
            self._save_test_result(res, test_dataset_id)
        else:
            # Compute the folds
            X_train_folds = np.array_split(train_df.sample(frac=1), self.number_of_folds)

            result = None

            for i in range(self.number_of_folds):
                local_train = pd.concat([X_train_folds[x] for x in range(self.number_of_folds) if x is not i])
                local_test = X_train_folds[i]

                res = compute(local_train, local_test, f"tweet_feature_engagement_is_{self._get_suffix()}")

                if result is None:
                    result = res
                else:
                    result = pd.concat([result, res])

            self._save_train_result_if_not_present(result, train_dataset_id)


class NumberOfEngagementsWithLanguagePositive(NumberOfEngagementsWithLanguageAbstract):

    def _get_suffix(self) -> str:
        return "positive"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsPositive(dataset_id=dataset_id)


class NumberOfEngagementsWithLanguageNegative(NumberOfEngagementsWithLanguageAbstract):

    def _get_suffix(self) -> str:
        return "negative"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsNegative(dataset_id=dataset_id)


class NumberOfEngagementsWithLanguageLike(NumberOfEngagementsWithLanguageAbstract):

    def _get_suffix(self) -> str:
        return "like"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsLike(dataset_id=dataset_id)


class NumberOfEngagementsWithLanguageRetweet(NumberOfEngagementsWithLanguageAbstract):

    def _get_suffix(self) -> str:
        return "retweet"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsRetweet(dataset_id=dataset_id)


class NumberOfEngagementsWithLanguageReply(NumberOfEngagementsWithLanguageAbstract):

    def _get_suffix(self) -> str:
        return "reply"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsReply(dataset_id=dataset_id)


class NumberOfEngagementsWithLanguageComment(NumberOfEngagementsWithLanguageAbstract):

    def _get_suffix(self) -> str:
        return "comment"

    def _get_engagement_feature(self, dataset_id) -> GeneratedFeaturePickle:
        return TweetFeatureEngagementIsComment(dataset_id=dataset_id)
