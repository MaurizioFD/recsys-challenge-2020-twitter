import numpy as np

from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set, \
    get_test_or_val_set_id_from_train
from Utils.Data.Dictionary.TweetTextFeaturesDictArray import TweetTokenLengthFeatureDictArray, \
    TweetTokenLengthUniqueFeatureDictArray
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId, MappedFeatureCreatorId, MappedFeatureTweetId


def find_ratio_and_update(creator_id, creator_length_array, creator_length_unique_array,
                          current_tweet_length, current_tweet_length_unique):

    # find the ratio
    if creator_length_array == 0:
        current_ratio = 0
    else:
        current_ratio = creator_length_unique_array[creator_id] / creator_length_array[creator_id]

    # update the arrays
    creator_length_array[creator_id] += current_tweet_length
    creator_length_unique_array[creator_id] += current_tweet_length_unique

    # return the result
    return current_ratio


class CreatorFrequencyUniqueTokens(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("creator_feature_frequency_of_unique_tokens", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creator_features/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/creator_features/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Check if the dataset id is train or test
        if is_test_or_val_set(self.dataset_id):
            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)
            test_dataset_id = get_test_or_val_set_id_from_train(train_dataset_id)
        else:
            train_dataset_id = self.dataset_id
            test_dataset_id = get_test_or_val_set_id_from_train(train_dataset_id)

        # Load features
        creation_timestamps_feature = RawFeatureTweetTimestamp(train_dataset_id)
        creators_feature = MappedFeatureCreatorId(train_dataset_id)
        tweet_id_feature = MappedFeatureTweetId(train_dataset_id)

        # Save the column name
        creators_col = creators_feature.feature_name
        tweet_id_col = tweet_id_feature.feature_name

        length_dict = TweetTokenLengthFeatureDictArray().load_or_create()
        length_unique_dict = TweetTokenLengthUniqueFeatureDictArray().load_or_create()

        dataframe = pd.concat([
            creators_feature.load_or_create(),
            creation_timestamps_feature.load_or_create(),
            tweet_id_feature.load_or_create()

        ], axis=1)

        dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

        creator_length_array = np.zeros(dataframe[creators_feature.feature_name].max() + 1, dtype=int)
        creator_length_unique_array = np.zeros(dataframe[creators_feature.feature_name].max() + 1, dtype=int)

        result = pd.DataFrame(
            [find_ratio_and_update(creator_id, creator_length_array, creator_length_unique_array,
                          length_dict[tweet_id], length_unique_dict[tweet_id])
             for creator_id, tweet_id in zip(dataframe[creators_col], dataframe[tweet_id_col])],
            index=dataframe.index
        )

        if not CreatorFrequencyUniqueTokens(train_dataset_id).has_feature():
            result.sort_index(inplace=True)
            CreatorFrequencyUniqueTokens(train_dataset_id).save_feature(result)

        if not CreatorFrequencyUniqueTokens(test_dataset_id).has_feature():

            # Load features
            creation_timestamps_feature = RawFeatureTweetTimestamp(test_dataset_id)
            creators_feature = MappedFeatureCreatorId(test_dataset_id)
            tweet_id_feature = MappedFeatureTweetId(test_dataset_id)

            # Save the column name
            creators_col = creators_feature.feature_name
            tweet_id_col = tweet_id_feature.feature_name

            dataframe = pd.concat([
                creation_timestamps_feature.load_or_create(),
                creators_feature.load_or_create(),
                tweet_id_feature.load_or_create(),
            ], axis=1)
            dataframe.sort_values(creation_timestamps_feature.feature_name, inplace=True)

            # if there are new creators in the test set, pad the arrays
            if dataframe[creators_col].max() + 1 > creator_length_array.size:
                creator_length_array = np.pad(
                    creator_length_array,
                    pad_width=(0, dataframe[creators_col].max() + 1 - creator_length_array.size),
                    mode='constant',
                    constant_values=0
                )

                creator_length_unique_array = np.pad(
                    creator_length_array,
                    pad_width=(0, dataframe[creators_col].max() + 1 - creator_length_unique_array.size),
                    mode='constant',
                    constant_values=0
                )


            result = pd.DataFrame(
                [find_ratio_and_update(creator_id, creator_length_array, creator_length_unique_array,
                                       length_dict[tweet_id], length_unique_dict[tweet_id])
                 for creator_id, tweet_id in zip(dataframe[creators_col], dataframe[tweet_id_col])],
                index=dataframe.index
            )

            result.sort_index(inplace=True)

            CreatorFrequencyUniqueTokens(test_dataset_id).save_feature(result)

