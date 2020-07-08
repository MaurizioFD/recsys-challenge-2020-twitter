
import pandas as pd


PATH = "tweet_tokens/subsample/val_text_tokens_days_2.csv"
RESULT_PATH = "tweet_tokens/subsample/val_text_tokens_days_2_unique.csv"

TWEET_ID = "tweet_features_tweet_id"
TWEET_TOKENS = "tweet_features_text_tokens"

current_index = 0

def save(df_row):
    global current_index
    if current_index % 1000000 == 0:
        print("Row : ", current_index)
    tweet_id = df_row[TWEET_ID]
    tokens = df_row[TWEET_TOKENS]
    #print(f"{tweet_id} and {tokens}")
    result_file.write(str(tweet_id) + ',' + tokens + '\n')
    current_index += 1

dataframe = pd.read_csv(PATH, header=0)

print(dataframe)

dataframe = dataframe.drop_duplicates(TWEET_ID)  #, inplace=True)

result_file = open(RESULT_PATH, "w+")
result_file.write(TWEET_ID + ',' + TWEET_TOKENS + '\n')

dataframe.apply(save, axis=1)

result_file.close()
