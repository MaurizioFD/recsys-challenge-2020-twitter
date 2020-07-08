
import time
import numpy as np

from TokenizerWrapper import TokenizerWrapper
from Utils import *


DEBUG = False

N_ROWS = 10

BERT_MODEL="models/bert-base-multilingual-cased"

# training
#BASE_PATH = "tweet_tokens/training/"
# evaluation
#BASE_PATH = "tweet_tokens/evaluation/"
# subsample 1 day
BASE_PATH="tweet_tokens/subsample/"

TWEET_ID = "tweet_features_tweet_id"
TWEET_TOKENS = "tweet_features_text_tokens"

# input file
TWEET_TOKENS_FILE = BASE_PATH + "val_text_tokens_days_2_unique.csv"

# output files
# !!!!!!!!!!!!!!!!! REMEMBER TO MODIFY THIS !!!!!!!!!!!!!!!!!
RESULT_PATH = BASE_PATH +  "val_text_tokens_clean_days_2_unique.csv"
MENTIONS_PATH = BASE_PATH + "mentions/val_mentions_day_2.csv"
HASHTAGS_PATH = BASE_PATH + "hashtags/val_hashtags_day_2.csv"
LINKS_PATH = BASE_PATH + "links/val_links_day_2.csv"
LENGTH_PATH = BASE_PATH + "tweets_length/val_tweet_tokens_lentgh_day_2.csv"

TWEET_IDS_MAPPING_PATH = "tweet_tokens/mappings/direct_tweet_id.json"
MENTIONS_MAPPING_PATH = "tweet_tokens/mappings/mentions/direct.json"
HASHTAGS_MAPPING_PATH = "tweet_tokens/mappings/hashtags/direct.json"
LINKS_MAPPING_PATH = "tweet_tokens/mappings/links/direct.json"


def main():

    # ignore header
    header = tokens_file.readline()
    
    #print("Header : ", header)

    finished = False
    row = 0

    start = time.time()
    
    while not finished:
        
        if row % 1000000 == 0:
            elapsed_time = time.time() - start
            print("Reading line : ", row, ' - Elapsed time: ', elapsed_time)
            
        if DEBUG:
            if row == N_ROWS:
                finished = True
            
        mentions_tokens = []
        hashtags_tokens = []

        line = str(tokens_file.readline())

        if line != '':

            line = line.replace("\\n",'').replace("\\t",'\t')
            
            if DEBUG:
                print(line)

            line = replace_escaped_chars(line)
            
            tweet_id, tokens_list = split_line(line)
                
            if DEBUG:
                print('\ntweet_id: ', tweet_id)
                print(tokens_list)
                decoded_tweet = tok.decode(tokens_list)
                print('\n', decoded_tweet, '\n')
                
            # limit the number of repeated special tokens (e.g. ">>>>>>>>>", "????????????", etc)
            tokens_list = reduce_num_special_tokens(tokens_list)
            
            tokens_list[-1] = tokens_list[-1].replace('\n','')
            
            # retweets contain the word RT (right after CLS, in position 1) followed
            # by mentions and then a ':', before starting with the actual tweet text
            if tokens_list[1] == special_tokens['RT'] and tokens_list[2] == special_tokens['@']:
                tokens_list, mentions_tokens = get_RT_mentions(tokens_list, mentions_tokens)

            # remove remaining mentions and hashtags
            tokens_list, mentions_tokens, hashtags_tokens = get_remove_mentions_hashtags(tok, tokens_list, mentions_tokens, hashtags_tokens)

            mentions_count = len(mentions_tokens)
            mentions_strings = decode_hashtags_mentions(tok, mentions_tokens)
            mapped_mentions = map_to_unique_ids(mentions_strings, mentions_dict, current_mapping_mentions)

            hashtags_count = len(hashtags_tokens)
            hashtags_strings = decode_hashtags_mentions(tok, hashtags_tokens)
            mapped_hashtags = map_to_unique_ids(hashtags_strings, hashtags_dict, current_mapping_hashtags)
            
            # remove links
            tokens_list, links_tokens, links_strings = get_remove_links(tok, tokens_list)
            links_count = len(links_tokens)
            
            mapped_links = map_to_unique_ids(links_strings, links_dict, current_mapping_links)
            
            if DEBUG:
                print('cleaned tokens: ', tokens_list)
                print('cleaned tweet: ', tok.decode(tokens_list))
                
                print('mentions tokens: ', mentions_tokens)
                print('mentions text: ', mentions_strings)
                print('mapped_mentions: ', mapped_mentions)
                print('mentions count: ', mentions_count)

                print('hashtag tokens: ', hashtags_tokens)
                print('hashtag text: ', hashtags_strings)
                print('mapped_hashtags: ', mapped_hashtags)
                print('hashtags count: ', hashtags_count)
            
                print('links: ', links_tokens)
                print('links text: ', links_strings)
                print('mapped links: ', mapped_links)
                print('links count: ', links_count)
            
            save_tweet(tweet_id, tokens_list, result_file)
            save(tweet_id, mentions_tokens, mentions_strings, mapped_mentions, mentions_count, mentions_file)
            save(tweet_id, hashtags_tokens, hashtags_strings, mapped_hashtags, hashtags_count, hashtags_file)
            save(tweet_id, links_tokens, links_strings, mapped_links, links_count, links_file)
            save_tweet_length(tweet_id, len(tokens_list)-2, length_file)  # -2 since all of them contain [CLS] and [SEP]

        else:
            finished = True
            print("\nEnd")

        row += 1
        
    print("Rows : ", row)
    print("Elapsed time : ", elapsed_time, '\n')
    print("\nDone.")
        
        
if __name__ == "__main__":
    
    global tok
    global tokens_file, result_file, mentions_file, hashtags_file, links_file, length_file
    global mentions_dict, current_mapping_mentions
    global hashtags_dict, current_mapping_hashtags
    global links_dict, current_mapping_links
    global tweet_ids_dict, current_mapping_tweet_ids
    
    tok = TokenizerWrapper(BERT_MODEL)
    
    # load mapping dictionaries
    #tweet_ids_dict, current_mapping_tweet_ids, _ = load_mapping(TWEET_IDS_MAPPING_PATH)
    mentions_dict, current_mapping_mentions, _ = load_mapping(MENTIONS_MAPPING_PATH)
    hashtags_dict, current_mapping_hashtags, _ = load_mapping(HASHTAGS_MAPPING_PATH)
    links_dict, current_mapping_links, _ = load_mapping(LINKS_MAPPING_PATH)
    
    # open necessary files
    tokens_file = open(TWEET_TOKENS_FILE, "r")  # input file
    result_file = open(RESULT_PATH, "w+")
    mentions_file = open(MENTIONS_PATH, "w+")
    hashtags_file = open(HASHTAGS_PATH, "w+")
    links_file = open(LINKS_PATH, "w+")
    length_file = open(LENGTH_PATH, "w+")
    
    # write header for resulting CSV files
    result_file.write(TWEET_ID + ',' + TWEET_TOKENS + "\n")
    mentions_file.write(TWEET_ID + "\x01mentions_count\x01mentions_tokens\x01mentions_text\x01mentions_mapped\n")
    hashtags_file.write(TWEET_ID + "\x01hashtags_count\x01hashtags_tokens\x01hashtags_text\x01hashtags_mapped\n")
    links_file.write(TWEET_ID + "\x01links_count\x01links_tokens\x01links_text\x01links_mapping\n")
    length_file.write(TWEET_ID + ",length\n")
    
    main()
    
    tokens_file.close()
    result_file.close()
    mentions_file.close()
    hashtags_file.close()
    links_file.close()
    length_file.close()
    
    # save updated mapping dictionaries
    #save_mapping('tweet_tokens/mentions_mapping_complete.json', mentions_dict)
    #save_mapping('tweet_tokens/hashtags_mapping_complete.json', hashtags_dict)
    #save_mapping('tweet_tokens/links_mapping_complete.json', links_dict)