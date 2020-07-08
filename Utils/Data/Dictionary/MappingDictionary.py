from abc import abstractmethod
from Utils.Data.Dictionary.Dictionary import Dictionary
import pathlib as pl
import pickle
import gzip
import json

from Utils.Data.Features.RawFeatures import *

from Utils.Text.TextUtils import *
from Utils.Text.TokenizerWrapper import TokenizerWrapper


class MappingDictionary(Dictionary):
    """
    Mapping dictionaries are built with the data of train and test set.
    """

    def __init__(self, dictionary_name: str, inverse: bool = False):
        super().__init__(dictionary_name)
        self.inverse = inverse
        self.direct_path_pck_path = pl.Path(f"{Dictionary.ROOT_PATH}/mapping/{self.dictionary_name}/direct.pck.gz")
        self.inverse_path_pck_path = pl.Path(f"{Dictionary.ROOT_PATH}/mapping/{self.dictionary_name}/inverse.pck.gz")
        self.direct_path_json_path = pl.Path(f"{Dictionary.ROOT_PATH}/mapping/{self.dictionary_name}/direct.json.gz")
        self.inverse_path_json_path = pl.Path(f"{Dictionary.ROOT_PATH}/mapping/{self.dictionary_name}/inverse.json.gz")

    def has_dictionary(self):
        if self.inverse:
            return self.inverse_path_pck_path.is_file()
        else:
            return self.direct_path_pck_path.is_file()

    def load_dictionary(self):
        if self.inverse:
            with gzip.GzipFile(self.inverse_path_pck_path, 'rb') as file:
                return pickle.load(file)
        else:
            with gzip.GzipFile(self.direct_path_pck_path, 'rb') as file:
                return pickle.load(file)

    @abstractmethod
    def create_dictionary(self):
        pass

    def save_dictionary(self, inverse_dictionary):
        dictionary = {v: k for k, v in inverse_dictionary.items()}
        self.direct_path_pck_path.parent.mkdir(parents=True, exist_ok=True)
        self.inverse_path_pck_path.parent.mkdir(parents=True, exist_ok=True)
        self.direct_path_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.inverse_path_json_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(self.direct_path_pck_path, 'wb') as file:
            pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.GzipFile(self.inverse_path_pck_path, 'wb') as file:
            pickle.dump(inverse_dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.GzipFile(self.direct_path_json_path, 'wb') as file:
            file.write(json.dumps(dictionary).encode('utf-8'))
        with gzip.GzipFile(self.inverse_path_json_path, 'wb') as file:
            file.write(json.dumps(inverse_dictionary).encode('utf-8'))


class MappingTweetIdDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_tweet_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetId("train")
        test_feature = RawFeatureTweetId("test")
        last_test_feature = RawFeatureTweetId("last_test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name],
            last_test_feature.load_or_create()[last_test_feature.feature_name]
        ])
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingUserIdDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_user_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature_creator = RawFeatureCreatorId("train")
        test_feature_creator = RawFeatureCreatorId("test")
        last_test_feature_creator = RawFeatureCreatorId("last_test")
        train_feature_engager = RawFeatureEngagerId("train")
        test_feature_engager = RawFeatureEngagerId("test")
        last_test_feature_engager = RawFeatureEngagerId("last_test")
        data = pd.concat([
            train_feature_creator.load_or_create()[train_feature_creator.feature_name],
            test_feature_creator.load_or_create()[test_feature_creator.feature_name],
            train_feature_engager.load_or_create()[train_feature_engager.feature_name],
            test_feature_engager.load_or_create()[test_feature_engager.feature_name],
            last_test_feature_creator.load_or_create()[last_test_feature_creator.feature_name],
            last_test_feature_engager.load_or_create()[last_test_feature_engager.feature_name]
        ])
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingLanguageDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_language_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetLanguage("train")
        test_feature = RawFeatureTweetLanguage("test")
        last_test_feature = RawFeatureTweetLanguage("last_test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name],
            last_test_feature.load_or_create()[last_test_feature.feature_name]
        ])
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingDomainDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_domain_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetDomains("train")
        test_feature = RawFeatureTweetDomains("test")
        last_test_feature = RawFeatureTweetDomains("last_test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name],
            last_test_feature.load_or_create()[last_test_feature.feature_name]
        ])
        data = pd.DataFrame([y for x in data.dropna() for y in x.split('\t')])
        data = data[data.columns[0]]
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingLinkDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_link_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetLinks("train")
        test_feature = RawFeatureTweetLinks("test")
        last_test_feature = RawFeatureTweetLinks("last_test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name],
            last_test_feature.load_or_create()[last_test_feature.feature_name]
        ])
        data = pd.DataFrame([y for x in data.dropna() for y in x.split('\t')])
        data = data[data.columns[0]]
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingHashtagDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_hashtag_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetHashtags("train")
        test_feature = RawFeatureTweetHashtags("test")
        last_test_feature = RawFeatureTweetHashtags("last_test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name],
            last_test_feature.load_or_create()[last_test_feature.feature_name]
        ])
        data = pd.DataFrame([y for x in data.dropna() for y in x.split('\t')])
        data = data[data.columns[0]]
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingMediaDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_media_id_dictionary", inverse)

    def create_dictionary(self):
        train_feature = RawFeatureTweetMedia("train")
        test_feature = RawFeatureTweetMedia("test")
        last_test_feature = RawFeatureTweetMedia("last_test")
        data = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name],
            last_test_feature.load_or_create()[last_test_feature.feature_name]
        ])
        data = pd.DataFrame([y for x in data.dropna() for y in x.split('\t')])
        data = data[data.columns[0]]
        dictionary = pd.DataFrame(data.unique()).to_dict()[0]

        self.save_dictionary(dictionary)


class MappingMentionsDictionary(MappingDictionary):

    def __init__(self, inverse: bool = False):
        super().__init__("mapping_mentions_id_dictionary", inverse)
        self.direct_path_csv_path = pl.Path(f"{Dictionary.ROOT_PATH}/mapping/{self.dictionary_name}/direct.csv.gz")
        self.mentions_ids_dict = {} 
        self.current_mapping_mentions = 0
        self.tok = None
        
    def get_mentions(self, tokens):
        #tokens = replace_escaped_chars(tokens)
        mentions_tokens = []
        
        tokens_list = tokens.split('\t')

        if tokens_list[1] == special_tokens['RT'] and tokens_list[2] == special_tokens['@']:
            tokens_list, mentions_tokens = get_RT_mentions(tokens_list, mentions_tokens)

        tokens_list, mentions_tokens, _ = get_remove_mentions_hashtags(self.tok, tokens_list, mentions_tokens, [])
        #mentions_count = len(mentions_tokens)
        mentions_strings = decode_hashtags_mentions(self.tok, mentions_tokens)
        mapped_mentions, self.current_mapping_mentions = map_to_unique_ids(mentions_strings, self.mentions_ids_dict, self.current_mapping_mentions)

        #for i in range(len(mentions_tokens)):
        #    mentions_tokens[i] = '\t'.join(map(str, mentions_tokens[i]))

        # each mention is separated by a ";"
        # each token in a mention is separated by a "\t"
        # each mapped mention is separated by a "\t"
        return '\t'.join(map(str, mapped_mentions)) # int(mentions_count), ';'.join(mentions_tokens), ';'.join(mentions_strings),

    def create_dictionary(self):
        from Utils.Data.Features.MappedFeatures import MappedFeatureTweetId
        
        self.tok = TokenizerWrapper("bert-base-multilingual-cased")
        
        train_feature = MappedFeatureTweetId("train")
        test_feature = MappedFeatureTweetId("test")
        last_test_feature = MappedFeatureTweetId("last_test")
        tweet_ids = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name],
            last_test_feature.load_or_create()[last_test_feature.feature_name]
        ])
        
        #print(tweet_ids)
        
        train_feature = RawFeatureTweetTextToken("train")
        test_feature = RawFeatureTweetTextToken("test")
        last_test_feature = RawFeatureTweetTextToken("last_test")
        text = pd.concat([
            train_feature.load_or_create()[train_feature.feature_name],
            test_feature.load_or_create()[test_feature.feature_name],
            last_test_feature.load_or_create()[last_test_feature.feature_name]
        ])
        
        #print(text)
        
        del train_feature
        del test_feature
        del last_test_feature
        
        data = pd.concat([tweet_ids, text], axis=1)
        
        #print(data)
        
        del tweet_ids
        del text
        
        tokens_df = pd.DataFrame(data.drop_duplicates('mapped_feature_tweet_id'))
        tokens_df = tokens_df.set_index('mapped_feature_tweet_id')
        tokens_df = tokens_df['raw_feature_tweet_text_token']
        
        #print(tokens_df)
        
        del data
        
        result = pd.DataFrame(columns=['mentions_mapped'])
        result['mentions_mapped'] = tokens_df.apply(self.get_mentions)
        
        del tokens_df
                                                         
        #result['mentions_count'] = result['mentions_mapped'].apply(lambda x: len(x.split('\t')))

        #print(result)

        self.save_dictionary(result)
        
    def has_dictionary(self):
        return self.direct_path_csv_path.is_file()
        
    def load_dictionary(self):
        return pd.read_csv(self.direct_path_csv_path, compression="gzip", sep="\x01", header=0, index_col=0)
    
    def save_dictionary(self, dictionary):
        self.direct_path_csv_path.parent.mkdir(parents=True, exist_ok=True)
        dictionary.to_csv(self.direct_path_csv_path, header=True, index=True, sep='\x01')