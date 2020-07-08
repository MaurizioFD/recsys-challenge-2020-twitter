
import numpy as np
import pandas as pd

import gzip

from PCA import IPCA



#DEVICE = "cpu"

CHUNK_SIZE = 5

BATCH_SIZE = 5

NUM_COMPONENTS = 2

files = ["tweet_tokens/embeddings/training_embeddings.csv.gz", "tweet_tokens/embeddings/training_embeddings_2.csv.gz"]
#PATH = "tweet_tokens/embeddings/day_1/embeddings_day_1_clean_unique_COMPLETE.csv"
OUTPUT_PATH = "tweet_tokens/embeddings/test_embeddings_PCA_" + str(NUM_COMPONENTS) + ".csv.gz"


'''
def read_text_embeddings(emb_file):
    if i == 0:
        f = open(path, 'r')
        text_embeddings = np.genfromtxt(emb_file, delimiter=",", skip_header=1, usecols=range(1,768), max_rows=100000)
    
    return text_embeddings
'''


def write_header():
    output_file = gzip.open(OUTPUT_PATH, "wt")
    output_file.write("tweet_features_tweet_id")
    for i in range(0, NUM_COMPONENTS):
        output_file.write(',embedding_' + str(i))
    output_file.write('\n')
    output_file.close()
    
    
def write_to_csv(df, embeddings):
    output_file = gzip.open(OUTPUT_PATH, "at")
    for i, row in enumerate(df.itertuples(index=False)):
        output_file.write(str(row[0])) # get the tweet id
        for j in range(0, NUM_COMPONENTS):
            output_file.write(',' + str(embeddings[i][j]))
        output_file.write('\n')
    output_file.close()


if __name__ == "__main__":

    inc_pca = IPCA(num_components=NUM_COMPONENTS)
    
    # incremental training phase
    print('\nTRAINING\n')
    i = 0
    for f in files:
        #print(f)
        for chunk in pd.read_csv(f, usecols=range(1,769), chunksize=CHUNK_SIZE, compression='gzip', nrows=10):
            #print(chunk)
            print("Chunk :", i)
            text_embeddings = np.array(chunk.values, dtype=np.float32)
            print("Training set :", text_embeddings.shape)
            #del chunk
            i += 1

            transformed_embeddings = inc_pca.fit(text_embeddings, batch_size=BATCH_SIZE)
            #transformed_embeddings, reconstructed_embeddings, loss = inc_pca.fit_transform(text_embeddings, batch_size=BATCH_SIZE, with_loss=True)

            #print("Transformed embeddings : ", transformed_embeddings.shape)
            #print("Reconstructed embeddings : ", reconstructed_embeddings.shape)
            #print("\nPCA projection loss :", loss, "\n")
    
    # produce embeddings with reduced dimensionality
    print('\nPRODUCING REDUCED EMBEDDINGS\n')
    
    write_header()  # column names
    
    i = 0
    for f in files:
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, compression='gzip', nrows=10):
            #print(chunk)
            print("Chunk :", i)
            text_embeddings = np.array(chunk.iloc[:,1:769].values, dtype=np.float32)
            i += 1

            transformed_embeddings = inc_pca.transform(text_embeddings)

            #print("Transformed embeddings : ", transformed_embeddings.shape)

            write_to_csv(chunk, transformed_embeddings)
        
    #emb_file.close()