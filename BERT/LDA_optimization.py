
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

PATH_HASHTAGS = 'tweet_tokens/training/hashtags/hashtags.csv'
PATH_MENTIONS = 'tweet_tokens/training/mentions/mentions.csv'

df_hashtags = pd.read_csv(PATH_HASHTAGS, header=0, delimiter='\x01')
df_mentions = pd.read_csv(PATH_MENTIONS, header=0, delimiter='\x01')

df_all = pd.concat([df_hashtags, df_mentions], axis=1)

del df_hashtags
del df_mentions

df = df_all.sample(frac=0.02)
df = df.reset_index()

del df_all

print("Number of samples :", len(df))

df = df.drop(columns=['tweet_features_tweet_id', 'hashtags_count', 'hashtags_tokens', 'hashtags_mapped', 'mentions_count', 'mentions_tokens', 'mentions_mapped'])
df = df.dropna(subset=['hashtags_text', 'mentions_text'], axis=0)
df['hashtags_text'] = df['hashtags_text'].map(lambda x : x.replace(' ', ''))
df['hashtags_text'] = df['hashtags_text'].map(lambda x : x.replace(';', ' '))
df['mentions_text'] = df['mentions_text'].map(lambda x : x.replace(' ', ''))
df['mentions_text'] = df['mentions_text'].map(lambda x : x.replace(';', ' '))

df['text'] = df['hashtags_text'] + ' ' + df['mentions_text']

for min_df in [1, 2, 5, 10]:
    for max_features in [25, 50, 100, 200]:
        
        print("\n---> min_df :", min_df, " - max_features :", max_features)
        
        vectorizer = CountVectorizer(max_df=0.95,
                                        min_df=min_df,
                                        max_features=max_features,
                                        lowercase=True)

        data_vectorized = vectorizer.fit_transform(df['text'].to_list())

        data_dense = data_vectorized.todense()
        print("Sparsicity: ", ((data_dense > 0).sum() / data_dense.size) * 100, "%")

        search_params = {
            'n_components': [10, 15, 20, 25, 30], 
            'learning_decay': [0.25, 0.5, 0.75, 0.9],
            #'max_iter': [10, 25, 50],
            #'learning_method': ['batch', 'online'],
            #'batch_szie': [128],
            #'learning_offset': [1.0, 10.0, 25.0, 50.0],
            #'perp_tol': [1e-3, 1e-2, 1e-1, 1],
            #'min_change_tol': [1e-3, 1e-2, 1e-1],
            #'n_jobs': [-1],
            #'verbose': [1],
            #'random_state': [42],
            #'evaluate_every': [1]
        }

        lda = LatentDirichletAllocation(max_iter=50, batch_size=128, n_jobs=-1)

        model = GridSearchCV(lda, 
                             verbose=10, 
                             n_jobs=-1,
                             param_grid=search_params,
                            )

        model.fit(data_vectorized)

        best_lda_model = model.best_estimator_
        print("Best mparams: ", model.best_params_)
        print("Best log-likelihood Score: ", model.best_score_)
        print("Model perplexity: ", best_lda_model.perplexity(data_vectorized))

        result = best_lda_model.transform(data_vectorized)
        result = pd.DataFrame(result)
        dominant_topic = np.argmax(result.values, axis=1)
        result['dominant_topic'] = dominant_topic
        print(result)

        print("\nNumber of elements of Topic != #0 :", (result['dominant_topic'] == 0).sum())

        for index, topic in enumerate(best_lda_model.components_):
            print(f'Top 15 words for Topic #{index}')
            print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-20:]])
            #print(f'Weights for Topic #{index}')
            #print(np.sort(topic))
            print('\n')
