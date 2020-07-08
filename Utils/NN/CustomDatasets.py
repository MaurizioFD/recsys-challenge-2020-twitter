import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Utils.NN.NNUtils import get_max_len_cap


class CustomTestDatasetCap(Dataset):
    def __init__(self, df_features: pd.DataFrame,
                 df_tokens_reader: pd.io.parsers.TextFileReader,
                 cap: int = 128):

        self.df_features = df_features
        self.df_tokens_reader_original = df_tokens_reader
        # self.df_tokens_reader_current = None
        self.count = -1
        self.cap = cap

    def __len__(self):
        return len(self.df_features)

    def __getitem__(self, index):

        # debug
        #print(f"debug-> index is:{index}")

        # if true, update the caches, i.e. self.tensors
        if index % self.df_tokens_reader_original.chunksize == 0:
            sep_tok_id = 102
            if index == 0:
                self.count = 0

            else:
                self.count += 1

            df_tokens_cache = self.df_tokens_reader_original.get_chunk()
            df_tokens_cache.columns = ['tokens']

            start = index
            end = start + self.df_tokens_reader_original.chunksize
            df_features_cache = self.df_features.iloc[start:end]

            text_series = df_tokens_cache['tokens'].map(lambda x: x.split('\t'))
            #print(f"first text_series: {text_series}")
            max_len, is_capped = get_max_len_cap(text_series, self.cap)
            attention_masks = np.ones((len(text_series), max_len), dtype=np.int8)
            if is_capped:
                debug_first_branch = False
                debug_second_branch = False

                # remove the additional tokens if exceeds max_len,
                # else: pad
                for i in range(len(text_series)):
                    debug_first_branch = False
                    debug_second_branch = False

                    i_shifted = i + index
                    if len(text_series[i_shifted]) > max_len:
                        debug_first_branch = True
                        # remove the additional tokens
                        while len(text_series[i_shifted]) >= (max_len):
                            text_series[i_shifted].pop()
                        # append the SEP token
                        text_series[i_shifted].append(sep_tok_id)

                    elif len(text_series[i_shifted]) < max_len:  # padding
                        debug_second_branch = True
                        initial_len = len(text_series[i_shifted])
                        miss = max_len - initial_len
                        text_series[i_shifted] += [0] * miss
                        for j in range(initial_len, max_len):
                            attention_masks[i][j] = 0
                    # print(
                    #    f"iteration {i}, debug_first_branch {debug_first_branch} ,debug_second_branch {debug_second_branch}, len: {len(text_series[i_shifted])}")
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)
                    # print(f"type of the array: {text_series[i_shifted].dtype}")

            else:  # if the series is not capped, normal padding

                # padding
                for i in range(len(text_series)):
                    i_shifted = i + index  # * self.df_tokens_reader.chunksize
                    initial_len = len(text_series[i_shifted])
                    miss = max_len - initial_len
                    text_series[i_shifted] += [0] * miss
                    for j in range(initial_len, max_len):
                        attention_masks[i][j] = 0
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)

            # todo we need to optimize this
            list_arr = []
            for feat in df_features_cache.columns:
                list_arr.append(df_features_cache[feat].values)
            feature_mat = np.array(list_arr)
            del list_arr
            gc.collect()

            text_series = text_series.map(lambda l: [int(elem) for elem in l]).map(
                lambda x: np.array(x, dtype=np.int32))

            # print(f"text_series : {text_series}")
            # print(f"text_series type: {type(text_series)}")
            # print(f"text_series to numpy: {text_series.to_numpy()}")

            text_np_mat = np.stack(text_series)
            #print(f"text_np_mat :\n {text_np_mat}")
            #print(f"text_np_mat shape :\n {text_np_mat.shape}")
            #print(f"text_np_mat type : {type(text_np_mat)}")
            #print(f"text_np_mat dtype : {text_np_mat.dtype}")
            #print(f"text_np_mat 0 type : {type(text_np_mat[0])}")

            #print(f"text_np_mat 0 : {text_np_mat[0]}")
            #print(f"text_np_mat 0 dtype : {text_np_mat[0].dtype}")
            text_tensor = torch.tensor(text_np_mat, dtype=torch.int64)
            attention_masks = torch.tensor(attention_masks, dtype=torch.int8)
            #print(df_label_cache['tweet_feature_engagement_is_like'])
            features = torch.tensor(feature_mat.T)
            self.tensors = [text_tensor, attention_masks, features]

        return tuple(tensor[index - self.count * self.df_tokens_reader_original.chunksize] for tensor in self.tensors)


class CustomDatasetCap(Dataset):
    def __init__(self, class_label : str,
                 df_features: pd.DataFrame,
                 df_tokens_reader: pd.io.parsers.TextFileReader,
                 df_label: pd.DataFrame,
                 cap: int = 128,
                 batches_to_skip: int = 0):

        self.df_features = df_features
        self.df_tokens_reader_original = df_tokens_reader
        self.df_tokens_reader_current = None
        self.df_label = df_label
        self.count = -1
        self.cap = cap
        self.class_label = class_label
        self.batches_to_skip = batches_to_skip

    def __len__(self):
        return len(self.df_features)

    def __getitem__(self, index):

        # debug
        #print(f"debug-> index is:{index}")

        # if true, update the caches, i.e. self.tensors
        if index % self.df_tokens_reader_original.chunksize == 0:
            sep_tok_id = 102
            if index == 0:
                self.count = 0
                self.df_tokens_reader_current = pd.read_csv(self.df_tokens_reader_original.f,
                                                            chunksize=self.df_tokens_reader_original.chunksize, index_col=0, header=0,)

                for j in range(0, self.batches_to_skip):
                    chunk = self.df_tokens_reader_current.get_chunk()

            else:
                self.count += 1

            df_tokens_cache = self.df_tokens_reader_current.get_chunk()
            df_tokens_cache.columns = ['tokens']

            start = index
            end = start + self.df_tokens_reader_current.chunksize
            df_features_cache = self.df_features.iloc[start:end]
            df_label_cache = self.df_label.iloc[start:end]

            df_tokens_cache.set_index(df_tokens_cache.index - self.batches_to_skip*self.df_tokens_reader_original.chunksize, inplace=True)

            #print(df_tokens_cache)
            #print(df_features_cache)

            text_series = df_tokens_cache['tokens'].map(lambda x: x.split('\t'))
            #print(f"first text_series: {text_series}")
            max_len, is_capped = get_max_len_cap(text_series, self.cap)
            attention_masks = np.ones((len(text_series), max_len), dtype=np.int8)
            if is_capped:
                debug_first_branch = False
                debug_second_branch = False

                # remove the additional tokens if exceeds max_len,
                # else: pad
                for i in range(len(text_series)):
                    debug_first_branch = False
                    debug_second_branch = False

                    i_shifted = i + index

                    #print("i ", i)
                    #print("index ", index)
                    #print("i_shifted ", i_shifted)

                    if len(text_series[i_shifted]) > max_len:
                        debug_first_branch = True
                        # remove the additional tokens
                        while len(text_series[i_shifted]) >= (max_len):
                            text_series[i_shifted].pop()
                        # append the SEP token
                        text_series[i_shifted].append(sep_tok_id)

                    elif len(text_series[i_shifted]) < max_len:  # padding
                        debug_second_branch = True
                        initial_len = len(text_series[i_shifted])
                        miss = max_len - initial_len
                        text_series[i_shifted] += [0] * miss
                        for j in range(initial_len, max_len):
                            attention_masks[i][j] = 0
                    # print(
                    #    f"iteration {i}, debug_first_branch {debug_first_branch} ,debug_second_branch {debug_second_branch}, len: {len(text_series[i_shifted])}")
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)
                    # print(f"type of the array: {text_series[i_shifted].dtype}")

            else:  # if the series is not capped, normal padding

                # padding
                for i in range(len(text_series)):
                    i_shifted = i + index  # * self.df_tokens_reader.chunksize
                    initial_len = len(text_series[i_shifted])
                    miss = max_len - initial_len
                    text_series[i_shifted] += [0] * miss
                    for j in range(initial_len, max_len):
                        attention_masks[i][j] = 0
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)

            # todo we need to optimize this
            list_arr = []
            for feat in df_features_cache.columns:
                list_arr.append(df_features_cache[feat].values)
            feature_mat = np.array(list_arr)
            del list_arr
            gc.collect()

            text_series = text_series.map(lambda l: [int(elem) for elem in l]).map(
                lambda x: np.array(x, dtype=np.int32))

            # print(f"text_series : {text_series}")
            # print(f"text_series type: {type(text_series)}")
            # print(f"text_series to numpy: {text_series.to_numpy()}")

            text_np_mat = np.stack(text_series)
            #print(f"text_np_mat :\n {text_np_mat}")
            #print(f"text_np_mat shape :\n {text_np_mat.shape}")
            #print(f"text_np_mat type : {type(text_np_mat)}")
            #print(f"text_np_mat dtype : {text_np_mat.dtype}")
            #print(f"text_np_mat 0 type : {type(text_np_mat[0])}")

            #print(f"text_np_mat 0 : {text_np_mat[0]}")
            #print(f"text_np_mat 0 dtype : {text_np_mat[0].dtype}")
            text_tensor = torch.tensor(text_np_mat, dtype=torch.int64)
            attention_masks = torch.tensor(attention_masks, dtype=torch.int8)
            #print(df_label_cache['tweet_feature_engagement_is_like'])
            labels = torch.tensor(df_label_cache[f'tweet_feature_engagement_is_{self.class_label}']
                                  .map(lambda x: 1 if x else 0).values, dtype=torch.int8)
            features = torch.tensor(feature_mat.T)
            self.tensors = [text_tensor, attention_masks, features, labels]

        return tuple(tensor[index - self.count * self.df_tokens_reader_current.chunksize] for tensor in self.tensors)


class CustomDatasetCapSubsample(Dataset):
    def __init__(self, class_label : str, 
                 df_features: pd.DataFrame,
                 df_tokens_reader: pd.io.parsers.TextFileReader,
                 df_label: pd.DataFrame,
                 cap: int = 128,
                 batch_subsample=None):

        self.df_features = df_features
        self.df_tokens_reader_original = df_tokens_reader
        self.df_tokens_reader_current = None
        self.df_label = df_label
        self.count = -1
        self.cap = cap
        self.batch_subsample = batch_subsample
        self.class_label = class_label

        #self.subsampled_batch_size = int(self.df_tokens_reader_original.chunksize * self.batch_subsample)

    def __len__(self):
        return int(len(self.df_features) * self.batch_subsample)

    def __getitem__(self, index):

        # debug
        #print(f"debug-> index is:{index}")

        current_batch_size = self.df_tokens_reader_original.chunksize
        current_subsampled_batch_size = int(current_batch_size * self.batch_subsample)

        #print("index :", index)

        # if true, update the caches, i.e. self.tensors
        if index % current_subsampled_batch_size == 0:
            sep_tok_id = 102
            if index == 0:
                self.count = 0
                self.df_tokens_reader_current = pd.read_csv(self.df_tokens_reader_original.f,
                                                            chunksize=self.df_tokens_reader_original.chunksize, index_col=0, header=0,)
            else:
                self.count += 1

            df_tokens_cache = self.df_tokens_reader_current.get_chunk()
            df_tokens_cache.columns = ['tokens']

            current_batch_size = len(df_tokens_cache)
            current_subsampled_batch_size = int(current_batch_size * self.batch_subsample)

            #print(current_batch_size, current_subsampled_batch_size)

            start = self.count * self.df_tokens_reader_current.chunksize
            end = start + current_batch_size

            #print(start, end)

            df_features_cache = self.df_features.iloc[start:end]
            df_label_cache = self.df_label.iloc[start:end]

            if self.batch_subsample is not None:
                mask = np.zeros(current_batch_size, dtype=int)
                mask[:current_subsampled_batch_size] = 1
                np.random.shuffle(mask)
                mask = mask.astype(bool)
                df_tokens_cache = df_tokens_cache[mask]
                df_features_cache = df_features_cache[mask]
                df_label_cache = df_label_cache[mask]
                #print(mask)
                #print(df_tokens_cache)
                #print(df_features_cache)
                #print(df_label_cache)
                new_index = pd.Series(range(index, index+current_subsampled_batch_size))
                df_tokens_cache.set_index(new_index, inplace=True)
                df_features_cache.set_index(new_index, inplace=True)
                df_label_cache.set_index(new_index, inplace=True)
                #print(df_tokens_cache)
                #print(df_features_cache)
                #print(df_label_cache)
            else:
                print("\nInvalid subsample value\n")
                return

            text_series = df_tokens_cache['tokens'].map(lambda x: x.split('\t'))
            #print(f"first text_series: {text_series}")
            max_len, is_capped = get_max_len_cap(text_series, self.cap)
            attention_masks = np.ones((len(text_series), max_len), dtype=np.int8)
            if is_capped:
                debug_first_branch = False
                debug_second_branch = False

                # remove the additional tokens if exceeds max_len,
                # else: pad
                for i in range(len(text_series)):
                    debug_first_branch = False
                    debug_second_branch = False

                    i_shifted = i + index
                    if len(text_series[i_shifted]) > max_len:
                        debug_first_branch = True
                        # remove the additional tokens
                        while len(text_series[i_shifted]) >= (max_len):
                            text_series[i_shifted].pop()
                        # append the SEP token
                        text_series[i_shifted].append(sep_tok_id)

                    elif len(text_series[i_shifted]) < max_len:  # padding
                        debug_second_branch = True
                        initial_len = len(text_series[i_shifted])
                        miss = max_len - initial_len
                        text_series[i_shifted] += [0] * miss
                        for j in range(initial_len, max_len):
                            attention_masks[i][j] = 0
                    # print(
                    #    f"iteration {i}, debug_first_branch {debug_first_branch} ,debug_second_branch {debug_second_branch}, len: {len(text_series[i_shifted])}")
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)
                    # print(f"type of the array: {text_series[i_shifted].dtype}")

            else:  # if the series is not capped, normal padding

                # padding
                for i in range(len(text_series)):
                    i_shifted = i + index  # * self.df_tokens_reader.chunksize
                    initial_len = len(text_series[i_shifted])
                    miss = max_len - initial_len
                    text_series[i_shifted] += [0] * miss
                    for j in range(initial_len, max_len):
                        attention_masks[i][j] = 0
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)

            # todo we need to optimize this
            list_arr = []
            for feat in df_features_cache.columns:
                list_arr.append(df_features_cache[feat].values)
            feature_mat = np.array(list_arr)
            del list_arr
            gc.collect()

            text_series = text_series.map(lambda l: [int(elem) for elem in l]).map(
                lambda x: np.array(x, dtype=np.int32))

            # print(f"text_series : {text_series}")
            # print(f"text_series type: {type(text_series)}")
            # print(f"text_series to numpy: {text_series.to_numpy()}")

            text_np_mat = np.stack(text_series)
            #print(f"text_np_mat :\n {text_np_mat}")
            #print(f"text_np_mat shape :\n {text_np_mat.shape}")
            #print(f"text_np_mat type : {type(text_np_mat)}")
            #print(f"text_np_mat dtype : {text_np_mat.dtype}")
            #print(f"text_np_mat 0 type : {type(text_np_mat[0])}")

            #print(f"text_np_mat 0 : {text_np_mat[0]}")
            #print(f"text_np_mat 0 dtype : {text_np_mat[0].dtype}")
            text_tensor = torch.tensor(text_np_mat, dtype=torch.int64)
            attention_masks = torch.tensor(attention_masks, dtype=torch.int8)
            #print(df_label_cache['tweet_feature_engagement_is_like'])
            labels = torch.tensor(df_label_cache[f'tweet_feature_engagement_is_{self.class_label}']
                                  .map(lambda x: 1 if x else 0).values, dtype=torch.int8)
            features = torch.tensor(feature_mat.T)
            self.tensors = [text_tensor, attention_masks, features, labels]

        return tuple(tensor[index - self.count * current_subsampled_batch_size] for tensor in self.tensors)


class CustomDatasetMultiCap(Dataset):
    def __init__(self,
                 df_features: pd.DataFrame,
                 df_tokens_reader: pd.io.parsers.TextFileReader,
                 df_label: pd.DataFrame,
                 cap: int = 128):

        self.df_features = df_features
        self.df_tokens_reader_original = df_tokens_reader
        self.df_tokens_reader_current = None
        self.df_label = df_label
        self.count = -1
        self.cap = cap

    def __len__(self):
        return len(self.df_features)

    def __getitem__(self, index):

        # debug
        #print(f"debug-> index is:{index}")

        # if true, update the caches, i.e. self.tensors
        if index % self.df_tokens_reader_original.chunksize == 0:
            sep_tok_id = 102
            if index == 0:
                self.count = 0
                self.df_tokens_reader_current = pd.read_csv(self.df_tokens_reader_original.f,
                                                            chunksize=self.df_tokens_reader_original.chunksize, index_col=0, header=0,)
            else:
                self.count += 1

            df_tokens_cache = self.df_tokens_reader_current.get_chunk()
            df_tokens_cache.columns = ['tokens']

            start = index
            end = start + self.df_tokens_reader_current.chunksize
            df_features_cache = self.df_features.iloc[start:end]
            df_label_cache = self.df_label.iloc[start:end]

            text_series = df_tokens_cache['tokens'].map(lambda x: x.split('\t'))
            #print(f"first text_series: {text_series}")
            max_len, is_capped = get_max_len_cap(text_series, self.cap)
            attention_masks = np.ones((len(text_series), max_len), dtype=np.int8)
            if is_capped:
                debug_first_branch = False
                debug_second_branch = False

                # remove the additional tokens if exceeds max_len,
                # else: pad
                for i in range(len(text_series)):
                    debug_first_branch = False
                    debug_second_branch = False

                    i_shifted = i + index
                    if len(text_series[i_shifted]) > max_len:
                        debug_first_branch = True
                        # remove the additional tokens
                        while len(text_series[i_shifted]) >= (max_len):
                            text_series[i_shifted].pop()
                        # append the SEP token
                        text_series[i_shifted].append(sep_tok_id)

                    elif len(text_series[i_shifted]) < max_len:  # padding
                        debug_second_branch = True
                        initial_len = len(text_series[i_shifted])
                        miss = max_len - initial_len
                        text_series[i_shifted] += [0] * miss
                        for j in range(initial_len, max_len):
                            attention_masks[i][j] = 0
                    # print(
                    #    f"iteration {i}, debug_first_branch {debug_first_branch} ,debug_second_branch {debug_second_branch}, len: {len(text_series[i_shifted])}")
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)
                    # print(f"type of the array: {text_series[i_shifted].dtype}")

            else:  # if the series is not capped, normal padding

                # padding
                for i in range(len(text_series)):
                    i_shifted = i + index  # * self.df_tokens_reader.chunksize
                    initial_len = len(text_series[i_shifted])
                    miss = max_len - initial_len
                    text_series[i_shifted] += [0] * miss
                    for j in range(initial_len, max_len):
                        attention_masks[i][j] = 0
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)

            # todo we need to optimize this
            list_arr = []
            for feat in df_features_cache.columns:
                list_arr.append(df_features_cache[feat].values)
            feature_mat = np.array(list_arr)
            del list_arr
            gc.collect()

            text_series = text_series.map(lambda l: [int(elem) for elem in l]).map(
                lambda x: np.array(x, dtype=np.int32))

            # print(f"text_series : {text_series}")
            # print(f"text_series type: {type(text_series)}")
            # print(f"text_series to numpy: {text_series.to_numpy()}")

            text_np_mat = np.stack(text_series)
            #print(f"text_np_mat :\n {text_np_mat}")
            #print(f"text_np_mat shape :\n {text_np_mat.shape}")
            #print(f"text_np_mat type : {type(text_np_mat)}")
            #print(f"text_np_mat dtype : {text_np_mat.dtype}")
            #print(f"text_np_mat 0 type : {type(text_np_mat[0])}")

            #print(f"text_np_mat 0 : {text_np_mat[0]}")
            #print(f"text_np_mat 0 dtype : {text_np_mat[0].dtype}")
            text_tensor = torch.tensor(text_np_mat, dtype=torch.int64)
            attention_masks = torch.tensor(attention_masks, dtype=torch.int8)
            #print(df_label_cache['tweet_feature_engagement_is_like'])

            like_arr = df_label_cache[f'tweet_feature_engagement_is_like'].astype(np.int8).values
            retweet_arr = df_label_cache[f'tweet_feature_engagement_is_retweet'].astype(np.int8).values
            reply_arr = df_label_cache[f'tweet_feature_engagement_is_reply'].astype(np.int8).values
            comment_arr = df_label_cache[f'tweet_feature_engagement_is_comment'].astype(np.int8).values

            labels_mat = np.vstack([like_arr, retweet_arr, reply_arr, comment_arr])

            labels = torch.tensor(labels_mat.T, dtype=torch.int8)

            features = torch.tensor(feature_mat.T)
            self.tensors = [text_tensor, attention_masks, features, labels]

        return tuple(tensor[index - self.count * self.df_tokens_reader_current.chunksize] for tensor in self.tensors)


class CustomDatasetDualCap(Dataset):
    def __init__(self,
                 df_features: pd.DataFrame,
                 df_tokens_reader: pd.io.parsers.TextFileReader,
                 df_label: pd.DataFrame,
                 cap: int = 128,
                 batches_to_skip: int = 0):

        self.df_features = df_features
        self.df_tokens_reader_original = df_tokens_reader
        self.df_tokens_reader_current = None
        self.df_label = df_label
        self.count = -1
        self.cap = cap
        self.batches_to_skip = batches_to_skip

    def __len__(self):
        return len(self.df_features)

    def __getitem__(self, index):

        # debug
        #print(f"debug-> index is:{index}")

        # if true, update the caches, i.e. self.tensors
        if index % self.df_tokens_reader_original.chunksize == 0:
            sep_tok_id = 102
            if index == 0:
                self.count = 0
                self.df_tokens_reader_current = pd.read_csv(self.df_tokens_reader_original.f,
                                                            chunksize=self.df_tokens_reader_original.chunksize, index_col=0, header=0,)

                for j in range(0, self.batches_to_skip):
                    chunk = self.df_tokens_reader_current.get_chunk()
            else:
                self.count += 1

            df_tokens_cache = self.df_tokens_reader_current.get_chunk()
            df_tokens_cache.columns = ['tokens']

            df_tokens_cache.set_index(df_tokens_cache.index - self.batches_to_skip*self.df_tokens_reader_original.chunksize, inplace=True)

            #print(df_tokens_cache)
            #print(df_features_cache)

            start = index
            end = start + self.df_tokens_reader_current.chunksize
            df_features_cache = self.df_features.iloc[start:end]
            df_label_cache = self.df_label.iloc[start:end]

            text_series = df_tokens_cache['tokens'].map(lambda x: x.split('\t'))
            #print(f"first text_series: {text_series}")
            max_len, is_capped = get_max_len_cap(text_series, self.cap)
            attention_masks = np.ones((len(text_series), max_len), dtype=np.int8)
            if is_capped:
                debug_first_branch = False
                debug_second_branch = False

                # remove the additional tokens if exceeds max_len,
                # else: pad
                for i in range(len(text_series)):
                    debug_first_branch = False
                    debug_second_branch = False

                    i_shifted = i + index
                    if len(text_series[i_shifted]) > max_len:
                        debug_first_branch = True
                        # remove the additional tokens
                        while len(text_series[i_shifted]) >= (max_len):
                            text_series[i_shifted].pop()
                        # append the SEP token
                        text_series[i_shifted].append(sep_tok_id)

                    elif len(text_series[i_shifted]) < max_len:  # padding
                        debug_second_branch = True
                        initial_len = len(text_series[i_shifted])
                        miss = max_len - initial_len
                        text_series[i_shifted] += [0] * miss
                        for j in range(initial_len, max_len):
                            attention_masks[i][j] = 0
                    # print(
                    #    f"iteration {i}, debug_first_branch {debug_first_branch} ,debug_second_branch {debug_second_branch}, len: {len(text_series[i_shifted])}")
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)
                    # print(f"type of the array: {text_series[i_shifted].dtype}")

            else:  # if the series is not capped, normal padding

                # padding
                for i in range(len(text_series)):
                    i_shifted = i + index  # * self.df_tokens_reader.chunksize
                    initial_len = len(text_series[i_shifted])
                    miss = max_len - initial_len
                    text_series[i_shifted] += [0] * miss
                    for j in range(initial_len, max_len):
                        attention_masks[i][j] = 0
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)

            # todo we need to optimize this
            list_arr = []
            for feat in df_features_cache.columns:
                list_arr.append(df_features_cache[feat].values)
            feature_mat = np.array(list_arr)
            del list_arr
            gc.collect()

            text_series = text_series.map(lambda l: [int(elem) for elem in l]).map(
                lambda x: np.array(x, dtype=np.int32))

            # print(f"text_series : {text_series}")
            # print(f"text_series type: {type(text_series)}")
            # print(f"text_series to numpy: {text_series.to_numpy()}")

            text_np_mat = np.stack(text_series)
            #print(f"text_np_mat :\n {text_np_mat}")
            #print(f"text_np_mat shape :\n {text_np_mat.shape}")
            #print(f"text_np_mat type : {type(text_np_mat)}")
            #print(f"text_np_mat dtype : {text_np_mat.dtype}")
            #print(f"text_np_mat 0 type : {type(text_np_mat[0])}")

            #print(f"text_np_mat 0 : {text_np_mat[0]}")
            #print(f"text_np_mat 0 dtype : {text_np_mat[0].dtype}")
            text_tensor = torch.tensor(text_np_mat, dtype=torch.int64)
            attention_masks = torch.tensor(attention_masks, dtype=torch.int8)

            df_label_cache.columns = [0, 1]
            arr_0 = df_label_cache[0].astype(np.int8).values
            arr_1 = df_label_cache[1].astype(np.int8).values

            labels_mat = np.vstack([arr_0, arr_1])

            labels = torch.tensor(labels_mat.T, dtype=torch.int8)

            features = torch.tensor(feature_mat.T)
            self.tensors = [text_tensor, attention_masks, features, labels]

        return tuple(tensor[index - self.count * self.df_tokens_reader_current.chunksize] for tensor in self.tensors)

