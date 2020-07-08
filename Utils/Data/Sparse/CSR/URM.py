from Utils.Data.DataStats import get_max_user_id, get_max_tweet_id
from Utils.Data.Sparse.CSR_SparseMatrix import CSR_SparseMatrix
import scipy.sparse as sps
import numpy as np
import pandas as pd


class URM(CSR_SparseMatrix):

    def __init__(self, df: pd.DataFrame):
        super().__init__("urm_csr_matrix")

        assert df.columns.shape[0] == 3, "The dataframe must have exactly three columns"
        assert 'mapped_feature_engager_id' in df.columns, "The dataframe must have mapped_feature_engager_id column"
        assert 'mapped_feature_tweet_id' in df.columns, "The dataframe must have mapped_feature_tweet_id column"
        assert 'engagement' in df.columns, "The dataframe must have engagement column"

        self.df = df

        self.max_user_id = get_max_user_id()
        self.max_tweet_id = get_max_tweet_id()

    def create_matrix(self):
        # creation of the urm
        # taking only the positive interactions
        # it could be interesting to take also the negative with a -1 value
        self.df = self.df[self.df['engagement']==True]


        # the matrix will be User X Tweet
        engager_ids_arr = self.df['mapped_feature_engager_id'].values
        tweet_ids_arr = self.df['mapped_feature_tweet_id'].values
        interactions_arr = np.array([1] * len(self.df))

        urm = sps.coo_matrix((interactions_arr, (engager_ids_arr, tweet_ids_arr)),
                             shape=(self.max_user_id, self.max_tweet_id)).tocsr()

        sps.save_npz('urm.npz', urm)

    def get_as_urm(self):
        self.df = self.df[self.df['engagement']==True]


        # the matrix will be User X Tweet
        engager_ids_arr = self.df['mapped_feature_engager_id'].values
        tweet_ids_arr = self.df['mapped_feature_tweet_id'].values
        interactions_arr = np.array([1] * len(self.df))

        return sps.coo_matrix((interactions_arr, (engager_ids_arr, tweet_ids_arr)),
                             shape=(self.max_user_id, self.max_tweet_id)).tocsr()