from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import *


class EngagerFeatureKnowTweetLanguage(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_know_tweet_language", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_language/{self.feature_name}.csv.gz")

    def create_feature(self):
        if is_test_or_val_set(self.dataset_id):

            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)

            # Load the necessary features
            creator_id_feature = MappedFeatureCreatorId(train_dataset_id)
            engager_id_feature = MappedFeatureEngagerId(train_dataset_id)
            language_id_feature = MappedFeatureTweetLanguage(train_dataset_id)
            engagement_feature = TweetFeatureEngagementIsLike(train_dataset_id)

            # Load the dataframes
            creator_id_df = creator_id_feature.load_or_create()
            engager_id_df = engager_id_feature.load_or_create()
            language_id_df = language_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Concatenate the dataframes
            dataframe = pd.concat([
                creator_id_df,
                engager_id_df,
                language_id_df,
                engagement_df
            ],
                axis=1
            )

            # Filter the negative interactions
            positive_dataframe = dataframe[dataframe[engagement_feature.feature_name]]

            # Let's compute the known language when the user is creator
            dictionary_creator_df = pd.DataFrame(positive_dataframe[[
                creator_id_feature.feature_name,
                language_id_feature.feature_name,
                engagement_feature.feature_name
            ]].groupby([creator_id_feature.feature_name, language_id_feature.feature_name]).first())

            dictionary_creator_df.columns = ['users']

            dictionary_creator = dictionary_creator_df.to_dict()['users']

            # Let's compute the known language when the user is engager
            dictionary_engager_df = pd.DataFrame(positive_dataframe[[
                engager_id_feature.feature_name,
                language_id_feature.feature_name,
                engagement_feature.feature_name
            ]].groupby([engager_id_feature.feature_name, language_id_feature.feature_name]).first())

            dictionary_engager_df.columns = ['users']

            dictionary_engager = dictionary_engager_df.to_dict()['users']

            # Merge the two dictionaries
            dictionary_user = {**dictionary_creator, **dictionary_engager}

            # Load the test information
            test_engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            test_tweet_langugage_feature = MappedFeatureTweetLanguage(self.dataset_id)
            test_engager_id_df = test_engager_id_feature.load_or_create()
            test_tweet_langugage_df = test_tweet_langugage_feature.load_or_create()

            test_dataframe = pd.concat([
                test_engager_id_df,
                test_tweet_langugage_df
            ],
                axis=1
            )

            # Apply the super duper dictionary
            result_df = pd.DataFrame(
                test_dataframe[[
                    engager_id_feature.feature_name,
                    language_id_feature.feature_name
                ]].apply(lambda x: dictionary_user.get((x[0], x[1]), False), axis=1))

            # Save back the dataframe
            self.save_feature(result_df)
        else:

            # Load the necessary features
            creator_id_feature = MappedFeatureCreatorId(self.dataset_id)
            engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            language_id_feature = MappedFeatureTweetLanguage(self.dataset_id)
            engagement_feature = TweetFeatureEngagementIsLike(self.dataset_id)

            # Load the dataframes
            creator_id_df = creator_id_feature.load_or_create()
            engager_id_df = engager_id_feature.load_or_create()
            language_id_df = language_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Concatenate the dataframes
            dataframe = pd.concat([
                creator_id_df,
                engager_id_df,
                language_id_df,
                engagement_df
            ],
                axis=1
            )

            # Filter the negative interactions
            positive_dataframe = dataframe[dataframe[engagement_feature.feature_name]]

            # Let's compute the known language when the user is creator
            dictionary_creator_df = pd.DataFrame(positive_dataframe[[
                creator_id_feature.feature_name,
                language_id_feature.feature_name,
                engagement_feature.feature_name
            ]].groupby([creator_id_feature.feature_name, language_id_feature.feature_name]).first())

            dictionary_creator_df.columns = ['users']

            dictionary_creator = dictionary_creator_df.to_dict()['users']

            # Let's compute the known language when the user is engager
            dictionary_engager_df = pd.DataFrame(positive_dataframe[[
                engager_id_feature.feature_name,
                language_id_feature.feature_name,
                engagement_feature.feature_name
            ]].groupby([engager_id_feature.feature_name, language_id_feature.feature_name]).first())

            dictionary_engager_df.columns = ['users']

            dictionary_engager = dictionary_engager_df.to_dict()['users']

            # Merge the two dictionaries
            dictionary_user = {**dictionary_creator, **dictionary_engager}

            # Apply the super duper dictionary
            result_df = pd.DataFrame(
                dataframe[[
                    engager_id_feature.feature_name,
                    language_id_feature.feature_name
                ]].apply(lambda x: dictionary_user.get((x[0], x[1]), False), axis=1))

            # Save back the dataframe
            self.save_feature(result_df)
