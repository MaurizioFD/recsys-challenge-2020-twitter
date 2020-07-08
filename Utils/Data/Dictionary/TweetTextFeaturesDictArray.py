import pandas as pd
import pathlib as pl
import numpy as np
import RootPath
from abc import abstractmethod
import os

from Utils.Data.Features.MappedFeatures import *
from Utils.Data.Dictionary.MappingDictionary import *


class TweetTextFeatureDictArrayNumpy(Dictionary):
    """
    It is built only using train and test set.
    Abstract class representing a dictionary array that works with numpy/pickle file.
    """

    def __init__(self, dictionary_name: str, ):
        super().__init__(dictionary_name)
        self.csv_path = pl.Path(f"{Dictionary.ROOT_PATH}/from_text_token/{self.dictionary_name}.csv.gz")
        self.npz_path = pl.Path(f"{Dictionary.ROOT_PATH}/text_features/{self.dictionary_name}.npz")

    def has_dictionary(self):
        return self.npz_path.is_file()

    def load_dictionary(self):
        assert self.has_dictionary(), f"The dictionary {self.dictionary_name} does not exists. Create it first."
        return np.load(self.npz_path, allow_pickle=True)['x']

    @abstractmethod
    def create_dictionary(self):
        pass

    def save_dictionary(self, arr: np.ndarray):
        self.npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.npz_path, x=arr)
        

class TweetTextEmbeddingsFeatureDictArray(TweetTextFeatureDictArrayNumpy):

    def __init__(self, dictionary_name : str):
        super().__init__(dictionary_name)

    def create_dictionary(self):
        # simply convert the embeddings dataframe to a numpy array (of arrays)
        # get the list of embeddings columns (can vary among different datasets)
        with gzip.open(self.csv_path, "rt") as reader:
            columns = reader.readline().strip().split(',')
        
        # this will be the final dataframe
        embeddings_feature_df = pd.DataFrame()
        # load the tweet id column
        embeddings_feature_df = pd.read_csv(self.csv_path, usecols=[columns[0]])
        for col in columns[1:]:
            # load one embedding column at a time
            embeddings_feature_df[col] = pd.read_csv(self.csv_path, usecols=[col])
            
        # convert to numpy all the columns except the tweet id
        arr = np.array(embeddings_feature_df.sort_values(by='tweet_features_tweet_id')[columns[1:]])
        
        self.save_dictionary(arr)
        
        
class TweetTextEmbeddingsPCA10FeatureDictArray(TweetTextEmbeddingsFeatureDictArray):

    def __init__(self):
        super().__init__("text_embeddings_PCA_10_feature_dict_array")
        
        
class TweetTextEmbeddingsPCA32FeatureDictArray(TweetTextEmbeddingsFeatureDictArray):

    def __init__(self):
        super().__init__("text_embeddings_PCA_32_feature_dict_array")
        

class TweetTextEmbeddingsHashtagsMentionsLDA15FeatureDictArray(TweetTextEmbeddingsFeatureDictArray):

    def __init__(self):
        super().__init__("text_embeddings_hashtags_mentions_LDA_15_feature_dict_array")
        

class TweetTextEmbeddingsHashtagsMentionsLDA20FeatureDictArray(TweetTextEmbeddingsFeatureDictArray):

    def __init__(self):
        super().__init__("text_embeddings_hashtags_mentions_LDA_20_feature_dict_array")


# TODO the csv file is inconsistent, don't use these dictionaries

class TweetTokenLengthFeatureDictArray(TweetTextFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("tweet_token_length_feature_dict_array")

    def create_dictionary(self):
        # TODO check the path
        # path to the unique tweet tokens
        dir_path = os.path.dirname(__file__)
        tweet_tokens_csv_path = pl.Path(f"{Dictionary.ROOT_PATH}/from_text_token/tweet_tokens_all_unique.csv")


        # load the tweet id, token_list dataframe
        tokens_feature_df_reader = pd.read_csv(tweet_tokens_csv_path, chunksize=250000)
        length_arr = None

        for chunk in tokens_feature_df_reader:
            arr = chunk['tweet_features_text_tokens']\
                .map(lambda x: x.split('\t'))\
                .map(lambda x: len(x) - 2)\
                .values

            if length_arr is None:
                length_arr = arr
            else:
                length_arr = np.hstack([length_arr, arr])

        self.save_dictionary(length_arr)


class TweetTokenLengthUniqueFeatureDictArray(TweetTextFeatureDictArrayNumpy):

    def __init__(self):
        super().__init__("tweet_token_length_unique_feature_dict_array")

    def create_dictionary(self):
        # TODO check the path
        # path to the unique tweet tokens
        dir_path = os.path.dirname(__file__)
        tweet_tokens_csv_path = pl.Path(f"{Dictionary.ROOT_PATH}/from_text_token/tweet_tokens_all_unique.csv")

        # load the tweet id, token_list dataframe
        tokens_feature_df_reader = pd.read_csv(tweet_tokens_csv_path, chunksize=250000)
        length_arr = None

        for chunk in tokens_feature_df_reader:
            arr = chunk['tweet_features_text_tokens'] \
                .map(lambda x: x.split('\t'))\
                .map(lambda x: set(x))      \
                .map(lambda x: len(x) - 2) \
                .values
            if length_arr is None:
                length_arr = arr
            else:
                length_arr = np.hstack([length_arr, arr])

        self.save_dictionary(length_arr)
