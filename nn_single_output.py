from Models.NN.NNRec import DistilBertRec
from Utils.Data.Data import get_dataset, get_feature, get_feature_reader
from Utils.Submission.Submission import create_submission_file
import numpy as np
import time
import sys

def main(class_label, model_id):

    feature_list_1 = [
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
        "tweet_feature_creation_timestamp_day_phase_shifted"
    ]

    feature_list_2 = [
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

    chunksize = 192

    train_dataset = "cherry_train"
    val_dataset = "cherry_val"

    print(f"Training model : {model_id}")
    print(f"Running on label : {class_label}")

    if class_label == "comment":
        feature_list = feature_list_1
        train_batches_number = 10000
    elif class_label == "reply":
        feature_list = feature_list_2
        train_batches_number = 20000

    n_data_train = chunksize * train_batches_number

    val_batches_number = 10000
    n_data_val = chunksize * val_batches_number

    print(f"n_data_train: {n_data_train}")
    print(f"n_data_val: {n_data_val}")

    print(f"train_dataset: {train_dataset}")
    print(f"val_dataset: {val_dataset}")

    feature_train_df = get_dataset(features=feature_list, dataset_id=train_dataset)
    #   feature_train_df, _ = train_test_split(feature_train_df, train_size=0.2)

    label_train_df = get_feature(feature_name=f"tweet_feature_engagement_is_{class_label}", dataset_id=train_dataset)

    text_train_reader_df = get_feature_reader(feature_name="raw_feature_tweet_text_token", dataset_id=train_dataset,
                                              chunksize=chunksize)

    #    label_train_df, _ = train_test_split(label_train_df, train_size=0.2)

    feature_val_df = get_dataset(features=feature_list, dataset_id=val_dataset)

    label_val_df = get_feature(feature_name=f"tweet_feature_engagement_is_{class_label}", dataset_id=val_dataset)

    text_val_reader_df = get_feature_reader(feature_name="raw_feature_tweet_text_token", dataset_id=val_dataset,
                                            chunksize=chunksize)

    if model_id == 1:
        feature_train_df = feature_train_df.head(n_data_train)
        label_train_df = label_train_df.head(n_data_train)
        feature_val_df = feature_val_df.head(n_data_val)
        label_val_df = label_val_df.head(n_data_val)
    elif model_id == 2:
        feature_train_df = feature_train_df.iloc[n_data_train:2*n_data_train]
        label_train_df = label_train_df.iloc[n_data_train:2*n_data_train]
        feature_val_df = feature_val_df.iloc[n_data_val:2*n_data_val]
        label_val_df = label_val_df.iloc[n_data_val:2*n_data_val]

    ffnn_params = {
        'hidden_size_1': 128,
        'hidden_size_2': 64,
        'hidden_dropout_prob_1': 0.5,
        'hidden_dropout_prob_2': 0.5
    }

    rec_params = {
        'epochs': 1,
        'weight_decay': 1e-5,
        'lr': 2e-5,
        'cap_length': 128,
        'ffnn_params': ffnn_params,
        'class_label': class_label
    }

    #print(f"ffnn_params: {ffnn_params}")
    print(f"bert_params: {rec_params}")

    rec = DistilBertRec(**rec_params)

    ###   TRAINING
    if model_id == 1:
        stats = rec.fit(df_train_features=feature_train_df,
                    df_train_tokens_reader=text_train_reader_df,
                    df_train_label=label_train_df,
                    df_val_features=feature_val_df,
                    df_val_tokens_reader=text_val_reader_df,
                    df_val_label=label_val_df,
                    save_filename=f"{class_label}_{model_id}",
                    cat_feature_set=set([]),
                    #subsample=0.1, # subsample percentage of each batch
                    #pretrained_model_dict_path="saved_models/saved_model_yj_like_0.0001_774_128_64_0.1_0.1_epoch_5"
                )
    elif model_id == 2:
        stats = rec.fit(df_train_features=feature_train_df,
                    df_train_tokens_reader=text_train_reader_df,
                    df_train_label=label_train_df,
                    df_val_features=feature_val_df,
                    df_val_tokens_reader=text_val_reader_df,
                    df_val_label=label_val_df,
                    save_filename=f"{class_label}_{model_id}",
                    cat_feature_set=set([]),
                    train_batches_to_skip=train_batches_number,
                    val_batches_to_skip=val_batches_number
                    #subsample=0.1, # subsample percentage of each batch
                    #pretrained_model_dict_path="saved_models/saved_model_yj_like_0.0001_774_128_64_0.1_0.1_epoch_5"
                ) 

    print("STATS: \n")
    print(stats)
    with open('stats.txt', 'w+') as f:
        for s in stats:
            f.write(str(s) + '\n')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
