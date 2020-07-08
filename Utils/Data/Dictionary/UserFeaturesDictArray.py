import numpy as np

from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Dictionary.UserBasicFeaturesDictArray import UserBasicFeatureDictArrayNumpy
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId, MappedFeatureCreatorId, \
    MappedFeatureTweetLanguage, MappedFeatureTweetId
from Utils.Data.Sparse.CSR_SparseMatrix import CSR_SparseMatrix
import numpy as np
import scipy.sparse as sps
import time
import billiard as mp


class MainLanguageUserBasicFeatureDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self, dataset_id):
        super().__init__("main_language_user_dict_array")
        self.dataset_id = dataset_id
        self.npz_path = pl.Path(f"{Dictionary.ROOT_PATH}/basic_features/user/{self.dataset_id}/{self.dictionary_name}.npz")

    def create_dictionary(self):
        # Load Language matrix
        csr_matrix = LanguageMatrix(self.dataset_id).load_or_create()

        # Cast it to dataframe
        df = pd.DataFrame.sparse.from_spmatrix(csr_matrix)

        # Save the base columns
        base_columns = df.columns

        # Find the main language
        df["main_language"] = df.idxmax(axis=1)

        # Find the amount of time a user has spoken a language
        df['total_count_language_occurence'] = df[base_columns].sum(axis=1)

        # Override the value for users that have never spoken a language
        df["main_language"] = [x if y > 0 else -1 for x, y in
                               zip(df["main_language"], df['total_count_language_occurence'])]

        # To create the some statistics
        # df.drop(columns=['total_count_language_occurence'])
        #
        # # To numpy matrix
        # matrix = df.groupby("main_language").sum()[1:].to_numpy(dtype=int)

        # Save the numpy matrix
        self.save_dictionary(df["main_language"].to_numpy())


class LanguageDictArray(UserBasicFeatureDictArrayNumpy):

    def __init__(self, dataset_id):
        super().__init__("language_user_dict_array")
        self.dataset_id = dataset_id
        self.npz_path = pl.Path(f"{Dictionary.ROOT_PATH}/basic_features/user/{self.dataset_id}/{self.dictionary_name}.npz")

    def create_dictionary(self):
        result = pd.DataFrame()
        if is_test_or_val_set(self.dataset_id):
            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)
            test_dataset_id = get_test_or_val_set_id_from_train(train_dataset_id)
        else:
            train_dataset_id = self.dataset_id
            test_dataset_id = get_test_or_val_set_id_from_train(train_dataset_id)

        # Load the index column
        train_engager_id_feature = MappedFeatureEngagerId(train_dataset_id)
        test_engager_id_feature = MappedFeatureEngagerId(test_dataset_id)
        train_creator_id_feature = MappedFeatureCreatorId(train_dataset_id)
        test_creator_id_feature = MappedFeatureCreatorId(test_dataset_id)
        train_language_id_feature = MappedFeatureTweetLanguage(train_dataset_id)
        test_language_id_feature = MappedFeatureTweetLanguage(test_dataset_id)
        train_is_positive_feature = TweetFeatureEngagementIsPositive(train_dataset_id)
        train_tweet_id_feature = MappedFeatureTweetId(train_dataset_id)
        test_tweet_id_feature = MappedFeatureTweetId(test_dataset_id)

        # Find the mask of uniques one
        engager_train_df = train_engager_id_feature.load_or_create()
        engager_test_df = test_engager_id_feature.load_or_create()
        creator_train_df = train_creator_id_feature.load_or_create()
        creator_test_df = test_creator_id_feature.load_or_create()
        is_positive_train_df = train_is_positive_feature.load_or_create()
        language_train_df = train_language_id_feature.load_or_create()
        language_test_df = test_language_id_feature.load_or_create()
        tweet_id_train_df = train_tweet_id_feature.load_or_create()
        tweet_id_test_df = test_tweet_id_feature.load_or_create()

        # Set index
        result['user'] = range(max(
            engager_train_df[train_engager_id_feature.feature_name].max(),
            engager_test_df[test_engager_id_feature.feature_name].max(),
            creator_train_df[train_creator_id_feature.feature_name].max(),
            creator_test_df[test_creator_id_feature.feature_name].max()
        ) + 1)
        result.set_index('user', inplace=True)

        # Create the creator dataframe
        creator_df = pd.concat(
            [
                creator_test_df.append(creator_train_df),
                language_test_df.append(language_train_df),
                tweet_id_test_df.append(tweet_id_train_df)
            ], axis=1
        ).drop_duplicates(train_tweet_id_feature.feature_name).drop(columns=train_tweet_id_feature.feature_name)

        creator_df.columns = ["user", "language"]

        # Create the engager dataframe
        engager_df = pd.concat(
            [
                engager_train_df[is_positive_train_df[train_is_positive_feature.feature_name]],
                language_train_df[is_positive_train_df[train_is_positive_feature.feature_name]]
            ], axis=1
        )

        engager_df.columns = ["user", "language"]

        dataframe = pd.concat(
            [
                creator_df,
                engager_df
            ]
        )

        # Group by and aggregate in numpy array
        result['language'] = dataframe.groupby("user").agg(list)['language'].apply(
            lambda x: np.array(x, dtype=np.uint8))
        result['language'].replace({np.nan: None}, inplace=True)

        # To numpy array
        arr = np.array(result['language'].array)

        self.save_dictionary(arr)


class LanguageMatrix(CSR_SparseMatrix):
    """
    Abstract class representing a feature in raw format that works with csv file.
    It is needed in order to cope with NAN/NA values.
    """

    def __init__(self, dataset_id):
        super().__init__("tweet_language_csr_matrix")
        self.dataset_id = dataset_id
        self.path = pl.Path(f"{CSR_SparseMatrix.ROOT_PATH}/sparse/{self.dataset_id}/{self.matrix_name}.npz")

    def create_matrix(self):
        nthread = 8
        nsplit = nthread * 100

        hashtag_dict = LanguageDictArray(self.dataset_id).load_or_create()

        chunks = np.array_split(hashtag_dict, nsplit)

        pairs = [None] * nsplit
        pairs[0] = (0, chunks[0])
        for i, chunk in enumerate(chunks):
            if i is not 0:
                pairs[i] = (sum([len(c) for c in chunks[:i]]), chunk)

        with mp.Pool(nthread) as p:
            results = p.map(_compute_on_sub_array, pairs)

        tweet_list = np.concatenate([r[0] for r in results])
        hashtag_list = np.concatenate([r[1] for r in results])
        data_list = np.concatenate([r[2] for r in results])

        csr_matrix = sps.csr_matrix(
            (data_list, (tweet_list, hashtag_list)),
            shape=(len(hashtag_dict), max(hashtag_list) + 1), dtype=np.uint32)

        self.save_matrix(csr_matrix)

def _compute_on_sub_array(tuple):
    offset = tuple[0]
    subarray = tuple[1]

    tweet_list = np.array([], dtype=np.uint32)
    hashtag_list = np.array([], dtype=np.uint32)

    for tweet_id, hashtags in enumerate(subarray):

        if hashtags is not None:
            hashtag_list = np.append(
                hashtag_list,
                hashtags
            )
            tweet_list = np.append(
                tweet_list,
                np.full(len(hashtags), tweet_id)
            )

    data_list = np.full(len(hashtag_list), 1, dtype=np.uint8)

    tweet_list = tweet_list + offset

    return tweet_list, hashtag_list, data_list
