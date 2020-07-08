
import time

from Utils import load_mapping

DEBUG = False

N_ROWS = 10

BERT_MODEL="models/distilbert-base-multilingual-cased"

# training
#BASE_PATH = "tweet_tokens/training/"
# evaluation
#BASE_PATH = "tweet_tokens/evaluation/"
# subsample 1 day
BASE_PATH="tweet_tokens/subsample/"

TWEET_ID = "tweet_features_tweet_id"
TWEET_TOKENS = "tweet_features_text_tokens"

# input file
TWEET_TOKENS_FILE = BASE_PATH + "val_days_2_raw.csv"

# output file
# !!!!!!!!!!!!!!!!! REMEMBER TO MODIFY THIS !!!!!!!!!!!!!!!!!
RESULT_PATH = BASE_PATH +  "val_text_tokens_days_2.csv"

TWEET_IDS_MAPPING_PATH = "tweet_tokens/mappings/direct_tweet_id.json"


def main():

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

        line = str(tokens_file.readline())

        if line != '':
            
            # column 0 contains text token, column 2 contains tweet id
            line = line.split('\x01', 3)[0:3] # drop other columns given in raw dataset
            line.pop(1)                       # drop hashtags list given in raw dataset

            tweet_id = str(tweet_ids_dict[line[1]])
            tokens = line[0].replace("\\n",'').replace("\\t",'\t')
            
            result_file.write(tweet_id + ',' + tokens + '\n')

        else:
            finished = True

        row += 1
        
    print("Rows : ", row)
    print("Elapsed time : ", elapsed_time, '\n')
    print("\nDone.")
        
        
if __name__ == "__main__":
    
    global tokens_file, result_file
    global tweet_ids_dict, current_mapping_tweet_ids
    
    # load mapping dictionaries
    tweet_ids_dict, current_mapping_tweet_ids, _ = load_mapping(TWEET_IDS_MAPPING_PATH)
    
    # open necessary files
    tokens_file = open(TWEET_TOKENS_FILE, "r")  # input file
    result_file = open(RESULT_PATH, "w+")
    
    # write header for resulting CSV files
    result_file.write(TWEET_ID + ',' + TWEET_TOKENS + "\n")
    
    main()
    
    tokens_file.close()
    result_file.close()