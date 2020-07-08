import functools

from tqdm.contrib.concurrent import process_map

from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Dictionary.TweetTextFeaturesDictArray import *
from Utils.Data.Features.Feature import Feature
from Utils.Data.Features.Generated.GeneratedFeature import GeneratedFeaturePickle
from Utils.Data.Features.MappedFeatures import MappedFeatureTweetId
from Utils.Data.Features.RawFeatures import RawFeatureTweetTextToken
import billiard as mp
from Utils.Text.TokenizerWrapper import TokenizerWrapper

from abc import abstractmethod
import pandas as pd
import numpy as np
import gzip
from tqdm import tqdm


def decode_tokens(row, tok):
    tokens = row.replace('\n', '').split('\t')
    sentence = tok.decode(tokens)
    return sentence

def process_decode_tokens_chunk(chunk):
    tok = TokenizerWrapper("bert-base-multilingual-cased")
    return chunk['raw_feature_tweet_text_token'].map(lambda x: decode_tokens(x, tok))

class TweetFeatureTextTokenDecoded(RawFeatureCSV):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_text_token_decoded", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

        self.tok = TokenizerWrapper("bert-base-multilingual-cased")

    def decode_tokens(self, row):
        tokens = row.replace('\n','').split('\t')
        sentence = self.tok.decode(tokens)
        return sentence
    
    def create_feature(self):
        import Utils.Data.Data as data
        # Load the tweet ids and tokens
        #tweet_tokens_feature = RawFeatureTweetTextToken(self.dataset_id)
        #tweet_tokens_df = tweet_tokens_feature.load_or_create()

        # load the tweet id, token_list dataframe
        tokens_feature_df_reader = data.get_feature_reader('raw_feature_tweet_text_token', self.dataset_id, chunksize=1000000)

        with mp.Pool(16) as p:
            decoded_tokens_arr = pd.concat([pd.concat(p.map(process_decode_tokens_chunk, np.array_split(chunk, 100))) for chunk in tokens_feature_df_reader])
        
        decoded_tokens_df = pd.DataFrame({'tweet_feature_text_token_decoded': decoded_tokens_arr})

        # Save the dataframe
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        decoded_tokens_df.to_csv(self.csv_path, compression='gzip', index=True)


class TweetFeatureMappedMentions(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_mentions", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):

        # Load tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()

        # Merge train and test mentions
        mentions_array = MappingMentionsDictionary().load_or_create()['mentions_mapped'].astype(str).map(
            lambda x: np.array(x.split('\t'), dtype=np.str) if x != 'nan' else None
        ).array

        # Compute for each engagement the tweet mentions
        mapped_mentions_df = pd.DataFrame(tweet_id_df[tweet_id_feature.feature_name].map(lambda x: mentions_array[x]))

        # Save the dataframe
        self.save_feature(mapped_mentions_df)

class TweetFeatureNumberOfMentions(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_number_of_mentions", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):
        # Load the extracted mentions
        mentions_feature = TweetFeatureMappedMentions(self.dataset_id)
        mentions_df = mentions_feature.load_or_create()

        # Compute for each engagement the tweet mentions
        mnumber_of_mentions_df = pd.DataFrame(mentions_df[mentions_feature.feature_name].map(lambda x: len(x) if x is not None else 0))

        # Save the dataframe
        self.save_feature(mnumber_of_mentions_df)
        

class TweetFeatureTextEmbeddings(GeneratedFeaturePickle):

    def __init__(self, feature_name: str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")
        self.embeddings_array = None
        
    @abstractmethod
    def load_embeddings_dictionary(self):
        pass

    def create_feature(self):
        # Load tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()
        
        #tweet_id_df = tweet_id_df.head(25)
        #print(tweet_id_df)
        
        self.embeddings_array = self.load_embeddings_dictionary()
        
        columns_num = self.embeddings_array.shape[1]
        
        # this will be the final dataframe
        embeddings_feature_df = pd.DataFrame()
        
        # for each column, map the embeddings dictionary to all the tweets
        for col in range(columns_num):
            print("column :", col)
            embeddings_feature_df[f"embedding_{col}"] = tweet_id_df["mapped_feature_tweet_id"].map(lambda x: self.embeddings_array[x, col])
            
        #print(embeddings_feature_df)
        
        # Save the dataframe
        self.save_feature(embeddings_feature_df)
        
        
class TweetFeatureTextEmbeddingsPCA32(TweetFeatureTextEmbeddings):

    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_clean_PCA_32", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsPCA32FeatureDictArray().load_or_create()
        

class TweetFeatureTextEmbeddingsPCA10(TweetFeatureTextEmbeddings):

    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_clean_PCA_10", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsPCA10FeatureDictArray().load_or_create()
        

class TweetFeatureTextEmbeddingsHashtagsMentionsLDA15(TweetFeatureTextEmbeddings):
        
    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_hashtags_mentions_LDA_15", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsHashtagsMentionsLDA15FeatureDictArray().load_or_create()
        

class TweetFeatureTextEmbeddingsHashtagsMentionsLDA20(TweetFeatureTextEmbeddings):
        
    def __init__(self, dataset_id: str):
        super().__init__("text_embeddings_hashtags_mentions_LDA_20", dataset_id)
        
    def load_embeddings_dictionary(self):
        self.embeddings_array = TweetTextEmbeddingsHashtagsMentionsLDA15FeatureDictArray().load_or_create()
        
    
class TweetFeatureDominantTopic(GeneratedFeaturePickle):

    def __init__(self, feature_name : str, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")
        self.dictionary_array = None
        
    @abstractmethod
    def load_dictionary(self):
        pass

    def create_feature(self):
        # Load the tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()
        
        self.dictionary_array = self.load_dictionary()
        
        df = pd.DataFrame()
        df["dominant_topic"] = tweet_id_df["mapped_feature_tweet_id"].map(lambda x: np.argmax(self.dictionary_array[x]) if np.max(self.dictionary_array[x]) == np.min(self.dictionary_array[x]) else -1)

        # Save the dataframe
        self.save_feature(df)
        

class TweetFeatureDominantTopicLDA15(TweetFeatureDominantTopic):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_dominant_topic_LDA_15", dataset_id)
        
    def load_dictionary(self):
        self.dictionary_array = TweetTextEmbeddingsHashtagsMentionsLDA15FeatureDictArray().load_or_create()
        
        
class TweetFeatureDominantTopicLDA20(TweetFeatureDominantTopic):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_dominant_topic_LDA_20", dataset_id)
        
    def load_dictionary(self):
        self.dictionary_array = TweetTextEmbeddingsHashtagsMentionsLDA20FeatureDictArray().load_or_create()
        

class TweetFeatureTokenLength(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_token_length", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):
        import Utils.Data.Data as data
        # Load the tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()

        # load the tweet id, token_list dataframe
        tokens_feature_df_reader = data.get_feature_reader('raw_feature_tweet_text_token', self.dataset_id, chunksize=250000)
        length_arr = None

        for chunk in tqdm(tokens_feature_df_reader):
            curr_arr = chunk['raw_feature_tweet_text_token'] \
                .map(lambda x: x.split('\t')) \
                .map(lambda x: len(x) - 2) \
                .values

            if length_arr is None:
                length_arr = curr_arr
            else:
                length_arr = np.hstack([length_arr, curr_arr])

        length_df = pd.DataFrame({'tweet_feature_token_length': length_arr})
        # Save the dataframe
        self.save_feature(length_df)


class TweetFeatureTokenLengthUnique(GeneratedFeaturePickle):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_token_length_unique", dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

    def create_feature(self):
        import Utils.Data.Data as data
        # Load the tweet ids
        tweet_id_feature = MappedFeatureTweetId(self.dataset_id)
        tweet_id_df = tweet_id_feature.load_or_create()

        # load the tweet id, token_list dataframe
        tokens_feature_df_reader = data.get_feature_reader('raw_feature_tweet_text_token', self.dataset_id, chunksize=250000)
        length_arr = None

        for chunk in tqdm(tokens_feature_df_reader):
            curr_arr = chunk['raw_feature_tweet_text_token'] \
                .map(lambda x: x.split('\t')) \
                .map(lambda x: set(x))\
                .map(lambda x: len(x) - 2) \
                .values

            if length_arr is None:
                length_arr = curr_arr
            else:
                length_arr = np.hstack([length_arr, curr_arr])

        length_df = pd.DataFrame({'tweet_feature_token_length_unique': length_arr})
        # Save the dataframe
        self.save_feature(length_df)


def count_contained_words(row, words_list):
    sentence = row.lower() # maybe remove spaces ???
    count = 0
    for w in words_list:
        if w in sentence:
            count += 1
    return count

def process_chunk(chunk, words_list):
    return chunk['tweet_feature_text_token_decoded'].map(lambda x: count_contained_words(x, words_list))

class TweetFeatureTextTopicWordCount(GeneratedFeaturePickle):

    def __init__(self, feature_name, dataset_id: str):
        super().__init__(feature_name, dataset_id)
        self.pck_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.pck.gz")
        self.csv_path = pl.Path(
            f"{Feature.ROOT_PATH}/{self.dataset_id}/generated/from_text_token/{self.feature_name}.csv.gz")

        self.words_list = []

    def count_contained_words(self, row):
        sentence = row.lower() # maybe remove spaces ???
        count = 0
        for w in self.words_list:
            if w in sentence:
                count += 1
        return count
    
    def create_feature(self):
        import Utils.Data.Data as data
        # Load the tweet ids and tokens
        #tweet_tokens_feature = TweetFeatureTextTokenDecoded(self.dataset_id)
        #tweet_tokens_df = tweet_tokens_feature.load_or_create()

        # load the tweet id, token_list dataframe
        tokens_feature_df_reader = data.get_feature_reader('tweet_feature_text_token_decoded', self.dataset_id, chunksize=1000000)

        with mp.Pool(16) as p:
            process_chunk_partial = functools.partial(process_chunk, words_list=self.words_list)
            word_count_arr = pd.concat([pd.concat(p.map(process_chunk_partial, np.array_split(chunk, 100))) for chunk in tokens_feature_df_reader])
        
        # words_count_df = pd.DataFrame(tweet_tokens_df['tweet_feature_text_token_decoded'].apply(self.count_contained_words))

        words_count_df = pd.DataFrame({'tweet_feature_text_token_decoded': word_count_arr})
        #print(words_count_df)
        print(f"Number of rows with {self.feature_name} == 0 :", (words_count_df['tweet_feature_text_token_decoded'] == 0).sum())
        
        # Save the dataframe
        self.save_feature(words_count_df)


class TweetFeatureTextTopicWordCountAdultContent(TweetFeatureTextTopicWordCount):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_text_topic_word_count_adult_content", dataset_id)
        
        self.words_list = ['adult content', 'adult film', 'adult movie', 'adult video', 'anal', 'ass', 'bara', 'barely legal', 'bdsm', 'bestiality', 'bisexual', 'bitch', 'blowjob', 'bondage', 'boob', 'boobs', 'boobies', 'boobys', 'booty', 'bound & gagged', 'bound and gagged', 'breast', 'breasts', 'bukkake', 'butt', 'cameltoe', 'creampie', 'cock', 'condom', 'cuck-old', 'cuckold', 'cum', 'cumshot', 'cunt', 'deep thraot', 'deap throat', 'deep thraoting', 'deap throating', 'deep-thraot', 'deap-throat', 'deep-thraoting', 'deap-throating', 'deepthraot', 'deapthroat', 'deepthraoting', 'deapthroating', 'dick', 'dildo', 'emetophilia', 'erotic', 'erotica', 'erection', 'erections', 'escort', 'facesitting', 'facial', 'felching', 'femdon', 'fetish', 'fisting', 'futanari', 'fuck', 'fucking', 'fucked', 'fucks', 'fucker', 'gangbang', 'gapping', 'gay', 'gentlemens club', 'gloryhole', 'glory hole', 'gonzo', 'gore', 'guro', 'handjob', 'hardon', 'hard-on', 'hentai', 'hermaphrodite', 'hidden camera', 'hump', 'humped', 'humping', 'hustler', 'incest', 'jerk off', 'jerking off', 'kinky', 'lesbian', 'lolicon ', 'masturbate', 'masturbating', 'masturbation', 'mature', 'mens club', 'menstrual', 'menstral', 'menstraul', 'milf', 'milking', 'naked', 'naughty', 'nude', 'orgasm', 'orgy', 'orgie', 'pearl necklace', 'pegging', 'penis', 'penetration', 'playboy', 'playguy', 'playgirl', 'porn', 'pornography', 'pornstar', 'pov', 'pregnant', 'preggo', 'pubic', 'pussy', 'rape', 'rimjob', 'scat', 'semen', 'sex', 'sexual', 'sexy', 'sexting', 'shemale', 'skank', 'slut', 'snuff', 'snuf', 'sperm', 'squirt', 'suck', 'swapping', 'tit', 'trans', 'transman', 'transsexual', 'transgender', 'threesome', 'tube8', 'twink', 'upskirt', 'vagina', 'virgin', 'whore', 'wore', 'xxx', 'yaoi', 'yif', 'yiff', 'yiffy', 'yuri', 'youporn']


class TweetFeatureTextTopicWordCountKpop(TweetFeatureTextTopicWordCount):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_text_topic_word_count_kpop", dataset_id)
        
        self.words_list = ['kpop', 'k pop', 'idol', 'comeback', 'mama', 'mnet', 'gda', 'goldendiscaward', 'golden disc award', '골든 디스크 시상식', '골든디스크시상식', 'gma', 'gaon', 'gaonmusicaward', 'gaon music award', 'music award', 'musicaward', 'nct', 'nct 127', 'nct127', 'bts', 'loona', '이달의소녀', 'gfriend', 'blackpink', 'exo', 'mcnd', 'monsta x', 'monstax', 'got7', 'mamamoo', 'twice', 'ateez', 'big bang', 'bigbang', 'red velvet', 'redvelvet', 'dkb', 'b.o.y', 'h&d', 'aoa', 'exid', 'iz*one', 'izone', 'itzy', 'cravity', 'btob', 'craxy', 'cignature', 'playm girls', 'playmgirls', '2z', 'clc', 'unvs', 'xenex', 'daydream', 'woo!ah!', 'pentagon', 'bandage', 'redsquare', '2nyne', 'trusty']


class TweetFeatureTextTopicWordCountCovid(TweetFeatureTextTopicWordCount):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_text_topic_word_count_covid", dataset_id)
        
        self.words_list = ['coronavirus', 'covid', 'covid19', 'covid-19', 'virus', 'syndrome', 'respiratory syndrome', 'severe acute respiratory syndrome', 'sars', 'sars 2', 'sars cov2', 'sars-cov2', 'middle east respiratory syndrome', 'mers', 'stay home', 'stayhome', 'distancing', 'socialdistancing', 'social distancing', 'epidemy', 'epidemic', 'pandemy', 'pandemic', 'emergency', 'state of emergency', 'lockdown', 'quarantine', 'self quarantine', 'self-quarantine', 'isolation', 'incubation', 'wuhan', 'china', 'chinavirus', 'china virus', 'bat', 'pangolin', 'pangolino', 'scaly anteaters', 'manis', 'respirator', 'intensive care unit', 'icu', 'fever', 'cough', 'shortness of breath', 'antibody', 'antibodies', 'symptoms', 'asymptomatic', 'world health organization', 'disease', 'containment', 'immunity', 'herd immunity', 'vaccine']


class TweetFeatureTextTopicWordCountSport(TweetFeatureTextTopicWordCount):

    def __init__(self, dataset_id: str):
        super().__init__("tweet_feature_text_topic_word_count_sport", dataset_id)
        
        self.words_list = ['sport', 'basketball', 'football', 'soccer', 'hockey', 'baseball', 'kobe', 'kobe bryant', 'kobebryant', 'rip kobe', 'ripkobe', 'nba', 'lakers', 'spurs', 'celtics', 'warriors', 'playoff', 'nba playoff', 'nbaplayoff', 'finals', 'nba finals', 'nbafinals', 'bball', 'lebron', 'lebron james', 'lebronjames', 'kingjames', 'nfl', 'nhl', 'mlb', 'epl', 'premiere', 'premiere league', 'premiereleague', 'seriea', 'serie a', 'liga', 'la liga', 'league', 'league1', 'league 1', 'bundesliga', 'efl', 'süper lig', 'super lig', 'süperlig', 'superlig', 'champions', 'champions league', 'espn', 'fox', 'foxnews', 'foxsport', 'sky sport', 'skysport', 'sport news', 'sportnews', 'sportsnews']