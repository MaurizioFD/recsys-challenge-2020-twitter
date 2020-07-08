import Utils.Data as data
from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Features.Feature import Feature
from Utils.Data.Features.Generated.EnsemblingFeature.MatrixEnsembling import ItemCBFMatrixEnsembling
from Utils.Data.Features.Generated.EnsemblingFeature.XGBEnsembling import XGBEnsembling
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
import pathlib as pl
import numpy as np
import pandas as pd
import hashlib

from Utils.Data.Sparse.CSR.CreatorTweetMatrix import CreatorTweetMatrix
from Utils.Data.Sparse.CSR.HashtagMatrix import HashtagMatrix
from Utils.Data.Sparse.CSR.URM import URM


class HashtagSimilarityFoldEnsembling(GeneratedFeaturePickle):

    def __init__(self,
                 dataset_id: str,
                 label: str,
                 number_of_folds: int = 5
                 ):
        feature_name = f"hashtag_similarity_fold_ensembling_{label}"
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/similarity_ensembling/{self.feature_name}.pck.gz")
        # self.csv_path = pl.Path(
        #     f"{Feature.ROOT_PATH}/{self.dataset_id}/similarity_ensembling/{self.feature_name}.csv.gz")
        # self.number_of_folds = number_of_folds
        # self.engager_features = [
        #     "mapped_feature_engager_id",
        #     "mapped_feature_tweet_id",
        #     f"tweet_feature_engagement_is_{label}"
        # ]
        # self.creator_features = [
        #     "mapped_feature_creator_id",
        #     "mapped_feature_tweet_id"
        # ]

    def create_feature(self):
        raise Exception("This feature is created externally. See gen_hashtag_similarity...py")
        # # Load the hashtag similarity
        # sim = HashtagMatrix().load_similarity().tocsr()
        #
        # # Check if the dataset id is train or test
        # if not is_test_or_val_set(self.dataset_id):
        #     # Compute train and test dataset ids
        #     train_dataset_id = self.dataset_id
        #
        #     # Load the dataset and shuffle it
        #     X_train = data.Data.get_dataset(features=self.engager_features,
        #                                     dataset_id=train_dataset_id).sample(frac=1)
        #
        #     creator_X_train = data.Data.get_dataset(features=self.creator_features,
        #                                             dataset_id=train_dataset_id)
        #
        #     # Create the ctm 'creator tweet matrix'
        #     ctm = CreatorTweetMatrix(creator_X_train).get_as_urm().astype(np.uint8)
        #
        #     # Compute the folds
        #     X_train_folds = np.array_split(X_train, self.number_of_folds)
        #
        #     # Declare list of scores (of each folds)
        #     # used for aggregating results
        #     scores = []
        #
        #     # Train multiple models with 1-fold out strategy
        #     for i in range(self.number_of_folds):
        #         # Compute the train set
        #         X_train = pd.concat([X_train_folds[x].copy() for x in range(self.number_of_folds) if x is not i])
        #         X_train.columns = [
        #             "mapped_feature_engager_id",
        #             "mapped_feature_tweet_id",
        #             "engagement"
        #         ]
        #
        #
        #         # Compute the test set
        #         X_test = X_train_folds[i].copy()
        #
        #         # Generate the dataset id for this fold
        #         fold_dataset_id = f"{self.feature_name}_{self.dataset_id}_fold_{i}"
        #
        #         # Load the urm
        #         urm = URM(X_train).get_as_urm().astype(np.uint8)
        #         urm = urm + ctm
        #
        #         # Create the sub-feature
        #         feature = ItemCBFMatrixEnsembling(self.feature_name, fold_dataset_id, urm, sim, X_train)
        #
        #         # Retrieve the scores
        #         scores.append(pd.DataFrame(feature.load_or_create()))
        #         print(X_test.index)
        #         print(scores.index)
        #
        #     # Compute the resulting dataframe and sort the results
        #     result = pd.concat(scores).sort_index()
        #
        #     # Save it as a feature
        #     self.save_feature(result)
        #
        # else:
        #     test_dataset_id = self.dataset_id
        #     train_dataset_id = get_train_set_id_from_test_or_val_set(test_dataset_id)
        #
        #     creator_X_train = data.Data.get_dataset(features=self.creator_features,
        #                                             dataset_id=train_dataset_id)
        #     creator_X_test = data.Data.get_dataset(features=self.creator_features,
        #                                             dataset_id=test_dataset_id)
        #     creator_X = pd.concat([creator_X_train, creator_X_test])
        #
        #     # Create the ctm 'creator tweet matrix'
        #     ctm = CreatorTweetMatrix(creator_X).get_as_urm().astype(np.uint8)
        #
        #     # Load the train dataset
        #     X_train = data.Data.get_dataset(features=self.engager_features, dataset_id=train_dataset_id)
        #     X_train.columns = [
        #         "mapped_feature_engager_id",
        #         "mapped_feature_tweet_id",
        #         "engagement"
        #     ]
        #     # Load the urm
        #     urm = URM(X_train).get_as_urm().astype(np.uint8)
        #     urm = urm + ctm
        #
        #     # Load the test dataset
        #     X_test = data.Data.get_dataset(features=self.engager_features, dataset_id=test_dataset_id)
        #     X_test.columns = ["user", "item", "engagement"]
        #
        #     # Create the sub-feature
        #     feature = ItemCBFMatrixEnsembling(self.feature_name, self.dataset_id, urm, sim, X_test.copy())
        #
        #     # Retrieve the scores
        #     result = pd.DataFrame(feature.load_or_create(), index=X_test.index)
        #
        #     # Save it as a feature
        #     self.save_feature(result)

class DomainSimilarityFoldEnsembling(GeneratedFeaturePickle):

    def __init__(self,
                 dataset_id: str,
                 label: str,
                 number_of_folds: int = 5
                 ):
        feature_name = f"domain_similarity_fold_ensembling_{label}"
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/similarity_ensembling/{self.feature_name}.pck.gz")


    def create_feature(self):
        raise Exception("This feature is created externally. See gen_hashtag_similarity...py")

class LinkSimilarityFoldEnsembling(GeneratedFeaturePickle):

    def __init__(self,
                 dataset_id: str,
                 label: str,
                 number_of_folds: int = 5
                 ):
        feature_name = f"link_similarity_fold_ensembling_{label}"
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/similarity_ensembling/{self.feature_name}.pck.gz")


    def create_feature(self):
        raise Exception("This feature is created externally. See gen_hashtag_similarity...py")