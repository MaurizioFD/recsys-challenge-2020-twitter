#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Utils.Base.Recommender_utils import check_matrix
from Utils.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender

from Utils.Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np

from Utils.Base.Similarity.Compute_Similarity import Compute_Similarity
from Utils.Base.Similarity.CosineSimilarity import cosine_similarity


class ItemKNNCFRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNN recommender"""


    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]



    def __init__(self, URM_train, verbose = True):
        super(ItemKNNCFRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        # similarity = Compute_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)
        self.W_sparse = cosine_similarity(self.URM_train.T, shrink=shrink, dense_output=False)


        self.W_sparse = check_matrix(self.W_sparse, format='csr')


    def get_prediction(self):
        pass


    def evaluate(self, URM_test):

        pass

