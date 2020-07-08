from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.MappedFeatures import MappedFeatureEngagerId


class EngagerFeatureKnowNumberOfLikeEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_known_number_of_like_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.csv.gz")

    def create_feature(self):
        if is_test_or_val_set(self.dataset_id):

            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)

            engager_id_feature = MappedFeatureEngagerId(train_dataset_id)
            engagement_feature = TweetFeatureEngagementIsLike(train_dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            test_engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            test_engager_id_df = test_engager_id_feature.load_or_create()

            engagement_count_df = pd.DataFrame(
                test_engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)
        else:

            engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            engagement_feature = TweetFeatureEngagementIsLike(self.dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            engagement_count_df = pd.DataFrame(
                engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)


class EngagerFeatureKnowNumberOfReplyEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_known_number_of_reply_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.csv.gz")

    def create_feature(self):
        if is_test_or_val_set(self.dataset_id):

            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)

            engager_id_feature = MappedFeatureEngagerId(train_dataset_id)
            engagement_feature = TweetFeatureEngagementIsReply(train_dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            test_engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            test_engager_id_df = test_engager_id_feature.load_or_create()

            engagement_count_df = pd.DataFrame(
                test_engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)
        else:

            engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            engagement_feature = TweetFeatureEngagementIsReply(self.dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            engagement_count_df = pd.DataFrame(
                engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)


class EngagerFeatureKnowNumberOfRetweetEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_known_number_of_retweet_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.csv.gz")

    def create_feature(self):
        if is_test_or_val_set(self.dataset_id):

            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)

            engager_id_feature = MappedFeatureEngagerId(train_dataset_id)
            engagement_feature = TweetFeatureEngagementIsRetweet(train_dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            test_engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            test_engager_id_df = test_engager_id_feature.load_or_create()

            engagement_count_df = pd.DataFrame(
                test_engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)
        else:

            engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            engagement_feature = TweetFeatureEngagementIsRetweet(self.dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            engagement_count_df = pd.DataFrame(
                engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)


class EngagerFeatureKnowNumberOfCommentEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_known_number_of_comment_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.csv.gz")

    def create_feature(self):
        if is_test_or_val_set(self.dataset_id):

            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)

            engager_id_feature = MappedFeatureEngagerId(train_dataset_id)
            engagement_feature = TweetFeatureEngagementIsComment(train_dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            test_engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            test_engager_id_df = test_engager_id_feature.load_or_create()

            engagement_count_df = pd.DataFrame(
                test_engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)
        else:

            engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            engagement_feature = TweetFeatureEngagementIsComment(self.dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            engagement_count_df = pd.DataFrame(
                engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)


class EngagerFeatureKnowNumberOfPositiveEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_known_number_of_positive_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.csv.gz")

    def create_feature(self):
        if is_test_or_val_set(self.dataset_id):

            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)

            engager_id_feature = MappedFeatureEngagerId(train_dataset_id)
            engagement_feature = TweetFeatureEngagementIsPositive(train_dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            test_engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            test_engager_id_df = test_engager_id_feature.load_or_create()

            engagement_count_df = pd.DataFrame(
                test_engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)
        else:

            engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            engagement_feature = TweetFeatureEngagementIsPositive(self.dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            engagement_count_df = pd.DataFrame(
                engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)


class EngagerFeatureKnowNumberOfNegativeEngagement(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("engager_feature_known_number_of_negative_engagement", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/know_engagement_count/{self.feature_name}.csv.gz")

    def create_feature(self):
        if is_test_or_val_set(self.dataset_id):

            train_dataset_id = get_train_set_id_from_test_or_val_set(self.dataset_id)

            engager_id_feature = MappedFeatureEngagerId(train_dataset_id)
            engagement_feature = TweetFeatureEngagementIsNegative(train_dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            test_engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            test_engager_id_df = test_engager_id_feature.load_or_create()

            engagement_count_df = pd.DataFrame(
                test_engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)
        else:

            engager_id_feature = MappedFeatureEngagerId(self.dataset_id)
            engagement_feature = TweetFeatureEngagementIsNegative(self.dataset_id)

            engager_id_df = engager_id_feature.load_or_create()
            engagement_df = engagement_feature.load_or_create()

            # Load the media column
            dataframe = pd.concat([
                engager_id_df,
                engagement_df,
            ],
                axis=1
            )
            dataframe = dataframe[dataframe[engagement_feature.feature_name]]
            dataframe = pd.DataFrame({self.feature_name: dataframe.groupby(engager_id_feature.feature_name).size()})
            dictionary = dataframe.to_dict()[self.feature_name]

            engagement_count_df = pd.DataFrame(
                engager_id_df[engager_id_feature.feature_name].map(lambda x: dictionary.get(x, 0)))
            self.save_feature(engagement_count_df)