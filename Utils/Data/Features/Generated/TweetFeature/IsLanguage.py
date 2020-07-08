from Utils.Data.DatasetUtils import is_test_or_val_set, get_train_set_id_from_test_or_val_set
from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle, GeneratedFeatureOnlyPickle
from Utils.Data.Features.MappedFeatures import MappedFeatureTweetLanguage


def top_popular_language(dataset_id: str, top_n: int = 5):
    # if is_test_or_val_set(dataset_id):
    #     dataset_id = get_train_set_id_from_test_or_val_set(dataset_id)
    #
    # dataframe = TweetFeatureIsLanguage(dataset_id).load_or_create()
    #
    # popularity_list = [(dataframe[column].sum(), dataframe[column]) for column in dataframe.columns]
    #
    # popularity_list = sorted(popularity_list, key=lambda x: x[0], reverse=True)
    #
    # selected_column = [tuple[1] for tuple in popularity_list][: top_n]
    #
    # selected_column_id = [col.name.split("_")[2] for col in selected_column]
    #
    # return selected_column_id

    return [0, 1, 2, 10]


class TweetFeatureIsLanguage(GeneratedFeatureOnlyPickle):

    def __init__(self, dataset_id: str, selected_languages: list = []):
        super().__init__("tweet_is_language_x", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/is_language/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/is_language/{self.feature_name}.csv.gz")
        self.selected_languages = selected_languages

    def load_feature(self):
        dataframe = super().load_feature()
        top_pop_list = []

        if len(self.selected_languages) > 0:
            selected_columns = ["is_language_" + str(language) for language in self.selected_languages]
            return dataframe[selected_columns]
        else:
            return dataframe

    def create_feature(self):
        # Load the languages
        languages = MappingLanguageDictionary().load_or_create().values()
        # Load the language column
        language_feature = MappedFeatureTweetLanguage(self.dataset_id)
        language_df = language_feature.load_or_create()
        # Creating the dataframe
        dataframe = pd.DataFrame()
        # Populating the dataframe
        for language in languages:
            dataframe[f"is_language_{language}"] = (language_df[language_feature.feature_name] == language)

        self.save_feature(dataframe)
