from Models.NN.NNRec import DistilBertRec
from Utils.Data.Data import get_dataset, get_feature, get_feature_reader
from Utils.Submission.Submission import create_submission_file
import numpy as np
import time
import sys
import pathlib
from Utils.TelegramBot import telegram_bot_send_update

def main(class_label, test_dataset, model_id):
    
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

    train_dataset = "cherry_train"

    print(f"Model : {model_id}")
    print(f"Running on label : {class_label}")

    training_chunksize = 192

    if class_label == "comment":
        feature_list = feature_list_1
        training_batches_number = 10000
        n_data_train = training_chunksize * training_batches_number
    elif class_label == "reply":
        feature_list = feature_list_2
        training_batches_number = 20000
        n_data_train = training_chunksize * training_batches_number

    ip = '34.242.41.76'
    submission_dir = f"Dataset/Features/{test_dataset}/ensembling"
    submission_filename = f"{submission_dir}/nn_predictions_{class_label}_{model_id}.csv"

    test_chunksize = 2048

    train_dataset = "cherry_train"

    print(f"Test dataset : {test_dataset}")

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

    saved_model_path = f"./saved_models/saved_model_{class_label}_{model_id}"

    rec = DistilBertRec(**rec_params)

    train_df = get_dataset(features=feature_list, dataset_id=train_dataset)

    if model_id == 1:
        train_df = train_df.head(n_data_train)
    elif model_id == 2:
        train_df = train_df.iloc[n_data_train:2*n_data_train]

    train_df = rec._normalize_features(train_df, is_train=True)
    
    ###   PREDICTION
    test_df = get_dataset(features=feature_list, dataset_id=test_dataset)
    #test_df = test_df.head(2500)

    prediction_start_time = time.time()

    text_test_reader_df = get_feature_reader(feature_name="raw_feature_tweet_text_token",
                                            dataset_id=test_dataset,
                                            chunksize=test_chunksize)

    predictions = rec.get_prediction(df_test_features=test_df,
                                     df_test_tokens_reader=text_test_reader_df,
                                     pretrained_model_dict_path=saved_model_path)
    print(f"Prediction time: {time.time() - prediction_start_time} seconds")

    print(predictions)
    print(predictions.shape)

    tweets = get_feature("raw_feature_tweet_id", test_dataset)["raw_feature_tweet_id"].array
    users = get_feature("raw_feature_engager_id", test_dataset)["raw_feature_engager_id"].array

    #tweets = tweets.head(2500).array
    #users = users.head(2500).array

    pathlib.Path(submission_dir).mkdir(parents=True, exist_ok=True)

    create_submission_file(tweets, users, predictions, submission_filename)

    #bot_string = f"DistilBertDoubleInput NN - {class_label} \n ---------------- \n"
    #bot_string = bot_string + f"@lucaconterio submission pronta! \nIP: {ip} \nFile: {submission_filename}"
    #telegram_bot_send_update(bot_string)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
