import functools

from tqdm.contrib.concurrent import process_map
import copy
from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
import pandas as pd
import numpy as np


def add(dictionary, key):
    dictionary[key] = dictionary.get(key, 0) + 1


def compute_chunk(chunk):
    timestamp = chunk.index.to_numpy().mean()
    dictionary = {}
    chunk['hashtags'].map(lambda x: [add(dictionary, e) for e in x] if x is not None else [0])
    return timestamp, dictionary


def get_popularity(chunk, result, s):
    out = []
    result = copy.deepcopy(result)
    s = copy.deepcopy(s)
    for hashtag, timestamp in zip(chunk['hashtags'], chunk['time']):
        if hashtag is not None:
            index = np.searchsorted(s, timestamp, 'left') - 1
            x = [result[index][1].get(h, 0)
                 for h in hashtag]
        else:
            x = [0]
        out.append(x)
    return pd.Series(out)

class HashtagPopularity(GeneratedFeaturePickle):

    def __init__(self, feature_name: str, dataset_id: str, window_size, window_overlap):
        super().__init__(feature_name, dataset_id)
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/hashtag_popularity/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/hashtag_popularity/{self.feature_name}.csv.gz")
        self.popularity_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/hashtag_popularity/{self.window_size}_{self.window_overlap}_popularity.npy")

    def get_popularity(self):
        import Utils.Data.Data as data
        if self.popularity_path.is_file():
            return np.load(self.popularity_path, allow_pickle=True)
        else:
            x = data.get_dataset(
                [
                    "mapped_feature_tweet_id",
                    "mapped_feature_tweet_hashtags",
                    "raw_feature_tweet_timestamp"
                ], self.dataset_id
            )

            x.columns = ["tweet", "hashtags", "time"]

            x = x.drop_duplicates("tweet")
            x = x.set_index('time', drop=True)
            x = x.sort_index()

            # Group size
            n = self.window_size
            # Overlapping size
            m = self.window_overlap

            chunks = [x[i:i + n] for i in range(0, len(x), n - m)]

            result = process_map(compute_chunk, chunks)
            s = [r[0] for r in result]
            y = data.get_dataset(
                [
                    "mapped_feature_tweet_id",
                    "mapped_feature_tweet_hashtags",
                    "raw_feature_tweet_timestamp"
                ], self.dataset_id
            )

            y.columns = ["tweet", "hashtags", "time"]
            get_popularity_partial = functools.partial(get_popularity, result=result, s=s)
            popularity = pd.concat(process_map(get_popularity_partial, np.array_split(y, 100)))
            self.popularity_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.popularity_path, popularity, allow_pickle=True)
            return popularity

class MaxHashtagPopularity(HashtagPopularity):

    def __init__(self, dataset_id: str, window_size, window_overlap):
        super().__init__(f"max_hashtag_popularity_{window_size}_{window_overlap}", dataset_id, window_size, window_overlap)

    def create_feature(self):
        popularity = self.get_popularity()
        max_popularity = np.array([max(p) for p in popularity])
        result = pd.DataFrame(max_popularity)
        self.save_feature(result)

class MinHashtagPopularity(HashtagPopularity):

    def __init__(self, dataset_id: str, window_size, window_overlap):
        super().__init__(f"min_hashtag_popularity_{window_size}_{window_overlap}", dataset_id, window_size, window_overlap)

    def create_feature(self):
        popularity = self.get_popularity()
        min_popularity = np.array([min(p) for p in popularity])
        result = pd.DataFrame(min_popularity)
        self.save_feature(result)

class MeanHashtagPopularity(HashtagPopularity):

    def __init__(self, dataset_id: str, window_size, window_overlap):
        super().__init__(f"mean_hashtag_popularity_{window_size}_{window_overlap}", dataset_id, window_size, window_overlap)

    def create_feature(self):
        popularity = self.get_popularity()
        mean_popularity = np.array([sum(p) / len(p) for p in popularity])
        result = pd.DataFrame(mean_popularity)
        self.save_feature(result)

class TotalHashtagPopularity(HashtagPopularity):

    def __init__(self, dataset_id: str, window_size, window_overlap):
        super().__init__(f"total_hashtag_popularity_{window_size}_{window_overlap}", dataset_id, window_size, window_overlap)

    def create_feature(self):
        popularity = self.get_popularity()
        total_popularity = np.array([sum(p) for p in popularity])
        result = pd.DataFrame(total_popularity)
        self.save_feature(result)
