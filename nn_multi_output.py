from Models.NN.MultiNNRec import MultiDistilBertRec
from Utils.Data.Data import get_dataset, get_feature, get_feature_reader
from Utils.Submission.Submission import create_submission_file
import numpy as np
import time
import pandas as pd


def main():
    '''
    feature_list = [
                "raw_feature_creator_follower_count",  # 0
                "raw_feature_creator_following_count",  # 1
                "raw_feature_engager_follower_count",  # 2
                "raw_feature_engager_following_count",  # 3
                "tweet_feature_number_of_photo",  # 4
                "tweet_feature_number_of_video",  # 5
                "tweet_feature_number_of_gif",  # 6
                "tweet_feature_number_of_hashtags",  # 7
                "tweet_feature_creation_timestamp_hour",  # 8
                "tweet_feature_creation_timestamp_week_day",  # 9
                "tweet_feature_number_of_mentions",  # 10
                "number_of_engagements_like", # 11
                "number_of_engagements_retweet", #  12
                "number_of_engagements_reply", # 13
                "number_of_engagements_comment", #  14
                "number_of_engagements_positive", #  15
                "number_of_engagements_negative", # 16
                "engager_feature_number_of_previous_like_engagement_ratio",  # 17
                "engager_feature_number_of_previous_reply_engagement_ratio",  # 18
                "engager_feature_number_of_previous_retweet_engagement_ratio",  # 19
                "engager_feature_number_of_previous_comment_engagement_ratio",  # 20
                "engager_feature_number_of_previous_positive_engagement_ratio",  # 21
                "engager_feature_number_of_previous_negative_engagement_ratio"  # 22
    ]
    '''
    '''
    feature_list = [
        "raw_feature_creator_follower_count",
        "raw_feature_creator_following_count",
        "raw_feature_engager_follower_count",
        "raw_feature_engager_following_count",
        "raw_feature_creator_is_verified",
        "raw_feature_engager_is_verified",
        "raw_feature_engagement_creator_follows_engager",
        "tweet_feature_number_of_photo",
        "tweet_feature_number_of_video",
        "tweet_feature_number_of_gif",
        "tweet_feature_number_of_media",
        "tweet_feature_is_retweet",
        "tweet_feature_is_quote",
        "tweet_feature_is_top_level",
        "tweet_feature_number_of_hashtags",
        "tweet_feature_creation_timestamp_hour",
        "tweet_feature_creation_timestamp_week_day",
        #"tweet_feature_number_of_mentions",
        "tweet_feature_token_length",
        "tweet_feature_token_length_unique",
        "tweet_feature_text_topic_word_count_adult_content",
        "tweet_feature_text_topic_word_count_kpop",
        "tweet_feature_text_topic_word_count_covid",
        "tweet_feature_text_topic_word_count_sport",
        "number_of_engagements_with_language_like",
        "number_of_engagements_with_language_retweet",
        "number_of_engagements_with_language_reply",
        "number_of_engagements_with_language_comment",
        "number_of_engagements_with_language_negative",
        "number_of_engagements_with_language_positive",
        "number_of_engagements_ratio_like",
        "number_of_engagements_ratio_retweet",
        "number_of_engagements_ratio_reply",
        "number_of_engagements_ratio_comment",
        "number_of_engagements_ratio_negative",
        "number_of_engagements_ratio_positive",
        "number_of_engagements_between_creator_and_engager_like",
        "number_of_engagements_between_creator_and_engager_retweet",
        "number_of_engagements_between_creator_and_engager_reply",
        "number_of_engagements_between_creator_and_engager_comment",
        "number_of_engagements_between_creator_and_engager_negative",
        "number_of_engagements_between_creator_and_engager_positive",
        "number_of_engagements_like",
        "number_of_engagements_retweet",
        "number_of_engagements_reply",
        "number_of_engagements_comment",
        "number_of_engagements_negative",
        "number_of_engagements_positive",
        "tweet_feature_creation_timestamp_hour_shifted",
        "tweet_feature_creation_timestamp_day_phase",
        "tweet_feature_creation_timestamp_day_phase_shifted",
        "engager_feature_number_of_previous_like_engagement_ratio",
        "engager_feature_number_of_previous_reply_engagement_ratio",
        "engager_feature_number_of_previous_retweet_engagement_ratio",
        "engager_feature_number_of_previous_comment_engagement_ratio",
        "engager_feature_number_of_previous_positive_engagement_ratio",
        "engager_feature_number_of_previous_negative_engagement_ratio",
        "adjacency_between_creator_and_engager_retweet",
        "adjacency_between_creator_and_engager_reply",
        "adjacency_between_creator_and_engager_comment",
        "adjacency_between_creator_and_engager_like",
        "adjacency_between_creator_and_engager_positive",
        "adjacency_between_creator_and_engager_negative",
        "graph_two_steps_adjacency_positive",
        "graph_two_steps_adjacency_negative",
        "graph_two_steps_adjacency_like",
        "graph_two_steps_adjacency_reply",
        "graph_two_steps_adjacency_retweet",
        "graph_two_steps_adjacency_comment",
        "graph_two_steps_positive",
        "graph_two_steps_negative",
        "graph_two_steps_like",
        "graph_two_steps_reply",
        "graph_two_steps_retweet",
        "graph_two_steps_comment"
    ]
    '''

    feature_list = [
        "raw_feature_creator_follower_count",  # 0
        "raw_feature_creator_following_count",  # 1
    ]

    chunksize = 192
    n_data_train = chunksize * 20000
    n_data_val = chunksize * 10000

    train_dataset = "cherry_train"
    val_dataset = "cherry_val"

    print("Running on labels : like - retweet - reply - comment")

    print(f"n_data_train: {n_data_train}")
    print(f"n_data_val: {n_data_val}")

    print(f"train_dataset: {train_dataset}")
    print(f"val_dataset: {val_dataset}")

    feature_train_df = get_dataset(features=feature_list, dataset_id=train_dataset)
    #   feature_train_df, _ = train_test_split(feature_train_df, train_size=0.2)
    feature_train_df = feature_train_df.head(n_data_train)

    like_df = get_feature(feature_name="tweet_feature_engagement_is_like", dataset_id=train_dataset)
    retweet_df = get_feature(feature_name="tweet_feature_engagement_is_retweet", dataset_id=train_dataset)
    reply_df = get_feature(feature_name="tweet_feature_engagement_is_reply", dataset_id=train_dataset)
    comment_df = get_feature(feature_name="tweet_feature_engagement_is_comment", dataset_id=train_dataset)
    label_train_df = pd.concat([like_df, retweet_df, reply_df, comment_df], axis=1)
    label_train_df = label_train_df.head(n_data_train)

    text_train_reader_df = get_feature_reader(feature_name="raw_feature_tweet_text_token", dataset_id=train_dataset,
                                              chunksize=chunksize)

    #    label_train_df, _ = train_test_split(label_train_df, train_size=0.2)

    feature_val_df = get_dataset(features=feature_list, dataset_id=val_dataset)
    feature_val_df = feature_val_df.head(n_data_val)

    like_df = get_feature(feature_name="tweet_feature_engagement_is_like", dataset_id=val_dataset)
    retweet_df = get_feature(feature_name="tweet_feature_engagement_is_retweet", dataset_id=val_dataset)
    reply_df = get_feature(feature_name="tweet_feature_engagement_is_reply", dataset_id=val_dataset)
    comment_df = get_feature(feature_name="tweet_feature_engagement_is_comment", dataset_id=val_dataset)
    label_val_df = pd.concat([like_df, retweet_df, reply_df, comment_df], axis=1)
    label_val_df = label_val_df.head(n_data_val)

    text_val_reader_df = get_feature_reader(feature_name="raw_feature_tweet_text_token", dataset_id=val_dataset,
                                            chunksize=chunksize)

    ffnn_params = {'hidden_size_1': 128, 'hidden_size_2': 64, 'hidden_dropout_prob_1': 0.5, 'hidden_dropout_prob_2': 0.5}
    rec_params = {'epochs': 5, 'weight_decay': 1e-5, 'lr': 2e-5, 'cap_length': 128, 'ffnn_params': ffnn_params}

    #print(f"ffnn_params: {ffnn_params}")
    print(f"bert_params: {rec_params}")

    rec = MultiDistilBertRec(**rec_params)

    ###   TRAINING
    stats = rec.fit(df_train_features=feature_train_df,
                df_train_tokens_reader=text_train_reader_df,
                df_train_label=label_train_df,
                df_val_features=feature_val_df,
                df_val_tokens_reader=text_val_reader_df,
                df_val_label=label_val_df,
                save_filename="multi_label"
                cat_feature_set=set([]),
                #subsample=0.1, # subsample percentage of each batch
                #pretrained_model_dict_path="saved_models/saved_model_yj_like_0.0001_774_128_64_0.1_0.1_epoch_5")
            )

    print("STATS: \n")
    print(stats)
    with open('stats.txt', 'w+') as f:
        for s in stats:
            f.write(str(s) + '\n')


if __name__ == '__main__':
    main()
