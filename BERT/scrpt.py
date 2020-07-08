
import time
import numpy as np

from TokenizerWrapper import TokenizerWrapper
from Utils import *


DEBUG = False

N_ROWS = 5

BERT_MODEL="models/distilbert-base-multilingual-cased"

# training
BASE_PATH = "tweet_tokens/training/"
# evaluation
#BASE_PATH = "tweet_tokens/evaluation/"
# subsample 1 day
#BASE_PATH="tweet_tokens/subsample/"

TWEET_ID = "tweet_features_tweet_id"
TWEET_TOKENS = "tweet_features_text_tokens"

MENTIONS_PATH = BASE_PATH + "mentions/mentions.csv"
HASHTAGS_PATH = BASE_PATH + "hashtags/hashtags.csv"
LINKS_PATH = BASE_PATH + "links/links.csv"


def main():

    # ignore header
    header = mentions_file.readline()
    header = TWEET_ID + ',' + header
    mentions_file_new.write(header)
    
    header = hashtags_file.readline()
    header = TWEET_ID + ',' + header
    hashtags_file_new.write(header)
    
    header = links_file.readline()
    header = TWEET_ID + ',' + header
    links_file_new.write(header)
    
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

        line = str(mentions_file.readline())

        if line != '':

            line = str(row) + ',' + line
            #print(line)
            mentions_file_new.write(line)
            
            line = str(hashtags_file.readline())
            line = str(row) + ',' + line
            hashtags_file_new.write(line)
            
            line = str(links_file.readline())
            line = str(row) + ',' + line
            links_file_new.write(line)

        else:
            finished = True

        row += 1
        
    print("Rows : ", row)
    print("Elapsed time : ", elapsed_time, '\n')
    print("\nDone.")
        
        
if __name__ == "__main__":
    
    global tok
    global mentions_file, hashtags_file, links_file
    
    # open necessary files
    mentions_file = open(MENTIONS_PATH, "r")
    hashtags_file = open(HASHTAGS_PATH, "r")
    links_file = open(LINKS_PATH, "r")
    
    mentions_file_new = open("mentions_new.csv", "w+")
    hashtags_file_new = open("hashtags_new.csv", "w+")
    links_file_new = open("links_new.csv", "w+")
    
    main()
    
    mentions_file.close()
    hashtags_file.close()
    links_file.close()
    
    mentions_file_new.close()
    hashtags_file_new.close()
    links_file_new.close()