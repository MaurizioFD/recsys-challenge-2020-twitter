import functools

from tqdm.contrib.concurrent import process_map

from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
import scipy.sparse as sps
import pandas as pd
import pathlib as pl
from Utils.Data.Features.Feature import Feature
from abc import ABC
from abc import abstractmethod
import numpy as np


def select_value_from_csr(matrix, row, cols):
    i_start, i_stop = matrix.indptr[row], matrix.indptr[row + 1]
    result = 0
    j_start = 0
    j_stop = len(cols) - 1
    i = i_start
    j = j_start
    while (True):
        if matrix.indices[i] < cols[j]:
            i += 1
        elif matrix.indices[i] > cols[j]:
            j += 1
        else:
            result += matrix.data[i]
            i += 1
            j += 1
        if i > i_stop or j > j_stop:
            break
    return result


def compute_score(dataframe, sim, user_array_dict):
    result = pd.DataFrame(
        [
            ((select_value_from_csr(sim, item, user_array_dict[user])) if user_array_dict[
                                                                              user] is not None else 0)
            for user, item
            in zip(dataframe["mapped_feature_engager_id"],
                   dataframe["mapped_feature_tweet_id"])
        ],
        index=dataframe.index
    )
    result.fillna(0, inplace=True)
    return result


def p_urm_chunks(urm, cold_user_dict):
    return [urm[i].indices if cold_user_dict[i] else None for i in range(urm.shape[0])]


class ItemCBFMatrixEnsembling(GeneratedFeaturePickle):

    def __init__(self,
                 feature_name: str,
                 dataset_id: str,
                 urm: sps.csr_matrix,
                 sim: sps.csr_matrix,
                 df_to_predict: pd.DataFrame):
        super().__init__(feature_name, dataset_id)

        assert urm.shape[1] == sim.shape[
            0], f"Check matrix format: urm shape is {urm.shape} and sim shape is {sim.shape}"
        assert "mapped_feature_engager_id" in df_to_predict.columns
        assert "mapped_feature_tweet_id" in df_to_predict.columns
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/similarity_item_cbf/{self.feature_name}.pck.gz")
        self.urm = urm
        self.sim = sim
        self.df_to_predict = df_to_predict

    def create_feature(self):
        # Loading the data
        urm = self.urm
        sim = self.sim
        dataframe = self.df_to_predict.copy()

        # Retrieving cold user and items
        cold_user_dict = np.full(urm.shape[0], True)
        cold_user_dict[urm.nonzero()[0]] = False
        cold_item_dict = np.full(sim.shape[0], True)
        cold_item_dict[sim.nonzero()[0]] = False

        # Filtering the dataframe
        dataframe['cold_user'] = dataframe["mapped_feature_engager_id"].map(lambda x: cold_user_dict[x])
        dataframe['cold_item'] = dataframe["mapped_feature_tweet_id"].map(lambda x: cold_item_dict[x])
        dataframe['cold'] = dataframe['cold_user'] | dataframe['cold_item']
        dataframe = dataframe[dataframe['cold'] == False]

        dataframe = dataframe[["mapped_feature_engager_id", "mapped_feature_tweet_id"]]

        # Creating auxiliary data structure
        idx = list(range(0, urm.shape[0], 1000000))
        idx.extend([urm.shape[0]])

        urm_chunks = []
        for i in range(len(idx) - 1):
            urm_chunks.append(urm[idx[i]:idx[i + 1]])

        partial_urm_chunks = functools.partial(p_urm_chunks, cold_user_dict=cold_user_dict)
        result = process_map(partial_urm_chunks, urm_chunks)
        user_array_dict = np.hstack(result)

        partial_compute_score = functools.partial(compute_score, sim=sim, user_array_dict=user_array_dict)
        chunks = np.array_split(dataframe, 100)
        result = process_map(partial_compute_score, chunks)
        result = pd.concat(result)

        result.fillna(0, inplace=True)
        print(result)
        self.save_feature(result)
