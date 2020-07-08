import functools

from tqdm import tqdm

from Utils.Data.DataUtils import FEATURES, DICTIONARIES, DICT_ARRAYS, SPARSE_MATRIXES
import pandas as pd
import numpy as np
import billiard as mp


def get_dataset_xgb(dataset_id: str = "train", X_label: list = None, Y_label: list = None):
    """
    :param dataset_id: The dataset id ("train", "test", etc.)
    :param X_label: The X features, the ones the model is trained on.
    :param Y_label:  The Y feature, the one to be predicted.
    :return: 2 dataframes: 1) X_Train, 2) Y_Train
    """
    if X_label is None:
        X_label = [
            "raw_feature_tweet_id",
            "raw_feature_engager_id"
        ]
    if Y_label is None:
        return get_dataset(X_label, dataset_id), None
    return get_dataset(X_label, dataset_id), get_dataset(Y_label, dataset_id)


def get_dataset_xgb_batch(total_n_split: int, split_n: int, dataset_id: str = "train", X_label: list = None,
                          Y_label: list = None, sample=1):
    """
    :param dataset_id: The dataset id ("train", "test", etc.)
    :param X_label: The X features, the ones the model is trained on.
    :param Y_label:  The Y feature, the one to be predicted.
    :return: 2 dataframes: 1) X_Train, 2) Y_Train
    """
    assert split_n < total_n_split, "split_n parameter should be less than total_n_split parameter"
    if X_label is None:
        X_label = [
            "raw_feature_tweet_id",
            "raw_feature_engager_id"
        ]
    if Y_label is None:
        return get_dataset_batch(X_label, dataset_id, total_n_split, split_n, sample), None

    return get_dataset_batch(X_label, dataset_id, total_n_split, split_n, sample), \
           get_dataset_batch(Y_label, dataset_id, total_n_split, split_n, sample)


def get_dataset_xgb_default_train():
    train_dataset = "train_split_with_timestamp_from_train_random_seed_888_timestamp_threshold_1581465600_holdout_10"
    # Define the X label
    X_label = [
        # "mapped_feature_tweet_id",
        # "mapped_feature_creator_id",
        # "mapped_feature_engager_id",
        "raw_feature_creator_follower_count",
        "raw_feature_creator_following_count",
        "raw_feature_engager_follower_count",
        "raw_feature_engager_following_count"
        "tweet_feature_number_of_photo",
        "tweet_feature_number_of_video",
        "tweet_feature_number_of_gif",
        "tweet_feature_number_of_media",
        "tweet_feature_number_of_hashtags",
        "tweet_feature_is_reply",
        "tweet_feature_is_retweet",
        "tweet_feature_is_quote",
        "tweet_feature_is_top_level",
        "engager_feature_know_tweet_language",
        "engager_feature_known_number_of_like_engagement",
        "engager_feature_known_number_of_reply_engagement",
        "engager_feature_known_number_of_retweet_engagement",
        "engager_feature_known_number_of_positive_engagement",
        "engager_feature_known_number_of_negative_engagement",
        "tweet_is_language_x"
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_comment"
    ]
    return get_dataset_xgb(dataset_id=train_dataset, X_label=X_label, Y_label=Y_label)


def get_dataset_xgb_default_test():
    train_dataset = "val_split_with_timestamp_from_train_random_seed_888_timestamp_threshold_1581465600_holdout_10"
    # Define the X label
    X_label = [
        # "mapped_feature_tweet_id",
        # "mapped_feature_creator_id",
        # "mapped_feature_engager_id",
        "raw_feature_creator_follower_count",
        "raw_feature_creator_following_count",
        "raw_feature_engager_follower_count",
        "raw_feature_engager_following_count"
        "tweet_feature_number_of_photo",
        "tweet_feature_number_of_video",
        "tweet_feature_number_of_gif",
        "tweet_feature_number_of_media",
        "tweet_feature_number_of_hashtags",
        "tweet_feature_is_reply",
        "tweet_feature_is_retweet",
        "tweet_feature_is_quote",
        "tweet_feature_is_top_level",
        "engager_feature_know_tweet_language",
        "engager_feature_known_number_of_like_engagement",
        "engager_feature_known_number_of_reply_engagement",
        "engager_feature_known_number_of_retweet_engagement",
        "engager_feature_known_number_of_positive_engagement",
        "engager_feature_known_number_of_negative_engagement",
        "tweet_is_language_x"
    ]
    # Define the Y label
    Y_label = [
        "tweet_feature_engagement_is_comment"
    ]
    return get_dataset_xgb(dataset_id=train_dataset, X_label=X_label, Y_label=Y_label)


def get_dataset(features: list, dataset_id: str, nthread:int=-1):
    all_features = features
    dataframe = pd.DataFrame()
    if nthread > 0:
        for features in np.array_split(all_features, int(len(all_features)/32) + 1):
            with mp.Pool(nthread) as p:
                partial_create_features = functools.partial(get_feature, dataset_id=dataset_id)
                # dataframe = pd.concat(p.map(partial_create_features, features), axis=1)
                dataframes = p.map(partial_create_features, features)
                if len(dataframe) > 0:
                    dataframes.append(dataframe)
                dataframe = pd.concat(dataframes, axis=1)
                for df in dataframes:
                    del df
                del dataframes
    else:
        for feature_name in tqdm(features):
            if (feature_name, dataset_id) in FEATURES.keys():
                f = FEATURES[(feature_name, dataset_id)]
                df = f.load_or_create()
                if len(df.columns) == 1:
                    dataframe[feature_name] = df[f.feature_name]
                else:
                    if len(dataframe) > 0:
                        dataframe = pd.concat([dataframe, df], axis=1)
                    else:
                        dataframe = df
            else:
                raise Exception(f"Feature {feature_name} not found ")
    # dataframe = pd.concat([get_feature(feature_name, dataset_id) for feature_name in features], axis=1)
    # Some columns are not in the format XGB expects, so the following block of code will cast them to the right format
    for column in dataframe.columns:
        if str(dataframe[column].dtype).lower()[:3] == "int":
            dataframe[column] = dataframe[column].fillna(0).astype(np.int64, copy=False)
        elif str(dataframe[column].dtype).lower() == "boolean":
            dataframe[column] = dataframe[column].fillna(False).astype(np.bool, copy=False)
    return dataframe


def get_dataset_batch(features: list, dataset_id: str, total_n_split: int, split_n: int, sample: float):
    assert split_n < total_n_split, "split_n parameter should be less than total_n_split parameter"

    if sample < 1:
        with mp.Pool(1) as p:
            partial_create_features = functools.partial(get_feature_batch, dataset_id=dataset_id,
                                                        total_n_split=total_n_split, split_n=split_n, sample=sample)
            dataframe = pd.concat(p.map(partial_create_features, features), axis=1)
    else:
        dataframe = pd.DataFrame()
        for feature_name in tqdm(features):
            if (feature_name, dataset_id) in FEATURES.keys():
                f = FEATURES[(feature_name, dataset_id)]
                df = np.array_split(f.load_or_create(), total_n_split)[split_n]
                if len(df.columns) == 1:
                    dataframe[feature_name] = df[f.feature_name]
                else:
                    if len(dataframe) > 0:
                        dataframe = pd.concat([dataframe, df], axis=1)
                    else:
                        dataframe = df
            else:
                raise Exception(f"Feature {feature_name} not found ")
        # dataframe = pd.concat([np.array_split(get_feature(feature_name, dataset_id),
        #                                       total_n_split)[split_n] for feature_name in features], axis=1)
    # Some columns are not in the format XGB expects, so the following block of code will cast them to the right format
    for column in dataframe.columns:
        if str(dataframe[column].dtype).lower()[:3] == "int":
            dataframe[column] = dataframe[column].fillna(0).astype(np.int64, copy=False)
        elif str(dataframe[column].dtype).lower() == "boolean":
            dataframe[column] = dataframe[column].fillna(False).astype(np.bool, copy=False)
    return dataframe


def get_feature(feature_name: str, dataset_id: str):
    if (feature_name, dataset_id) in FEATURES.keys():
        df = FEATURES[(feature_name, dataset_id)].load_or_create()
        if len(df.columns) == 1:
            df.columns = [feature_name]
        return df
    else:
        raise Exception(f"Feature {feature_name} not found ")


def get_feature_reader(feature_name: str, dataset_id: str, chunksize: int = 1000):
    if (feature_name, dataset_id) in FEATURES.keys():
        reader = FEATURES[(feature_name, dataset_id)].load_feature_reader(chunksize=chunksize)
        return reader
    else:
        raise Exception(f"Feature {feature_name} not found ")

def get_feature_batch(feature_name: str, dataset_id: str, total_n_split: int, split_n: int, sample: float):
    if (feature_name, dataset_id) in FEATURES.keys():
        df = np.array_split(get_feature(feature_name, dataset_id).sample(frac=sample, random_state=0),
                                              total_n_split)[split_n]
        if len(df.columns) == 1:
            df.columns = [feature_name]
        return df
    else:
        raise Exception(f"Feature {feature_name} not found ")


def get_dictionary(dictionary_name: str):
    if dictionary_name in DICTIONARIES.keys():
        return DICTIONARIES[dictionary_name].load_or_create()


def get_dictionary_array(dictionary_name: str):
    if dictionary_name in DICT_ARRAYS.keys():
        return DICT_ARRAYS[dictionary_name].load_or_create()


def get_csr_matrix(matrix_name: str):
    if matrix_name in SPARSE_MATRIXES.keys():
        return SPARSE_MATRIXES[matrix_name].load_or_create()

def oversample(dataframe: pd.DataFrame, column_name: str, value, desired_percentage: float):
    assert desired_percentage <= 1 and desired_percentage >= 0, "The desired percentage should be between 0 and 1"

    current_size = len(dataframe)

    sample = dataframe[dataframe[column_name] == value]
    current_selected_row_size = len(sample)

    current_percentage = current_selected_row_size / current_size
    delta_percentage = desired_percentage - current_percentage

    print(f"current_percentage: {current_percentage}, desired_percentage: {desired_percentage}, delta_percentage: {delta_percentage}")

    if delta_percentage > 0:

        assert delta_percentage > 0, "Something went wrong with oversampling, is the desired percentage high enough?"

        n_new_rows = int(delta_percentage * current_size)

        sample = sample.sample(n_new_rows, replace=True)
        dataframe = pd.concat([dataframe, sample])
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe

    else:

        return dataframe
