import functools
import pathlib
import multiprocessing
from Utils.Data.Dictionary.TweetBasicFeaturesDictArray import *
from Utils.Data.Dictionary.UserBasicFeaturesDictArray import *
from Utils.Data.Dictionary.TweetTextFeaturesDictArray import *
from Utils.Data.Features.Generated.CreatorFeature.CreatorFrequencyUniqueTokens import CreatorFrequencyUniqueTokens
from Utils.Data.Features.Generated.CreatorFeature.CreatorNumberOfEngagementGiven import \
    CreatorNumberOfEngagementGivenLike, CreatorNumberOfEngagementGivenReply, CreatorNumberOfEngagementGivenRetweet, \
    CreatorNumberOfEngagementGivenComment, CreatorNumberOfEngagementGivenPositive, \
    CreatorNumberOfEngagementGivenNegative
from Utils.Data.Features.Generated.CreatorFeature.CreatorNumberOfEngagementReceived import *
from Utils.Data.Features.Generated.CreatorFeature.CreatorNumberOfPreviousEngagementGiven import *
from Utils.Data.Features.Generated.CreatorFeature.CreatorNumberOfPreviousEngagementReceived import *
from Utils.Data.Features.Generated.EngagerFeature.AdjacencyBetweenCreatorAndEngager import *
from Utils.Data.Features.Generated.EngagerFeature.EngagerKnowTweetLanguage import *
from Utils.Data.Features.Generated.EngagerFeature.EngagerKnowsHashtag import *
from Utils.Data.Features.Generated.EngagerFeature.EngagerNumberOfEngagementsReceived import *
from Utils.Data.Features.Generated.EngagerFeature.GraphTwoSteps import *
from Utils.Data.Features.Generated.EngagerFeature.GraphTwoStepsAdjacency import *
from Utils.Data.Features.Generated.EngagerFeature.KnownEngagementCount import *
from Utils.Data.Features.Generated.EngagerFeature.NumberOfEngagements import *
from Utils.Data.Features.Generated.EngagerFeature.NumberOfEngagementsBetweenCreatorAndEngager import *
from Utils.Data.Features.Generated.EngagerFeature.NumberOfEngagementsRatio import *
from Utils.Data.Features.Generated.EngagerFeature.NumberOfEngagementsWithLanguage import *
from Utils.Data.Features.Generated.EngagerFeature.NumberOfPreviousEngagementBetweenCreatorAndEngager import *
from Utils.Data.Features.Generated.EngagerFeature.NumberOfPreviousEngagementRatio import *
from Utils.Data.Features.Generated.EngagerFeature.NumberOfPreviousEngagementWithLanguage import *
from Utils.Data.Features.Generated.EngagerFeature.NumberOfPreviousEngagements import *
from Utils.Data.Features.Generated.EnsemblingFeature.LGBMFoldEnsembling import LGBMFoldEnsemblingLike1
from Utils.Data.Features.Generated.EnsemblingFeature.SimilarityFoldEnsembling import *
from Utils.Data.Features.Generated.EnsemblingFeature.XGBFoldEnsembling import XGBFoldEnsemblingLike1, \
    XGBFoldEnsemblingRetweet1, XGBFoldEnsemblingReply1, XGBFoldEnsemblingComment1, XGBFoldEnsemblingLike2, \
    XGBFoldEnsemblingRetweet2, XGBFoldEnsemblingReply2, XGBFoldEnsemblingComment2
from Utils.Data.Features.Generated.LanguageFeature.MainGroupedLanguageFeature import *
from Utils.Data.Features.Generated.LanguageFeature.MainLanguageFeature import *
from Utils.Data.Features.Generated.TweetFeature.CreationTimestamp import *
from Utils.Data.Features.Generated.TweetFeature.FromTextToken import *
from Utils.Data.Features.Generated.TweetFeature.HashtagPopularity import *
from Utils.Data.Features.Generated.TweetFeature.IsEngagementType import *
from Utils.Data.Features.Generated.TweetFeature.IsLanguage import *
from Utils.Data.Features.Generated.TweetFeature.IsTweetType import *
from Utils.Data.Features.Generated.TweetFeature.NumberOfHashtags import TweetFeatureNumberOfHashtags
from Utils.Data.Features.Generated.TweetFeature.NumberOfMedia import *
from Utils.Data.Features.Generated.TweetFeature.HasDiscriminantHashtag import *
from Utils.Data.Features.Generated.TweetFeature.TweetNumberOfEngagements import *
from Utils.Data.Features.Generated.TweetFeature.TweetNumberOfPreviousEngagements import *
from Utils.Data.Features.MappedFeatures import *
from Utils.Data.Dictionary.MappingDictionary import *
from Utils.Data.Features.RawFeatures import *
from Utils.Data.Sparse.CSR.HashtagMatrix import *
from Utils.Data.Sparse.CSR.DomainMatrix import *
from Utils.Data.Sparse.CSR.Language.LanguageMatrixOnlyPositive import LanguageMatrixOnlyPositive
from Utils.Data.Sparse.CSR.LinkMatrix import *
import xgboost as xgb
import sklearn.datasets as skd
import billiard as mp
import os

DATASET_IDS = [
    "train",
    # Removing those datasets to speed up the generation of features
    # "train_days_1",
    # "train_days_12",
    # "train_days_123",
    # "train_days_1234",
    # "train_days_12345",
    # "train_days_123456",
    "test",
    # Removing those datasets to speed up the generation of features
    # "val_days_2",
    # "val_days_3",
    # "val_days_4",
    # "val_days_5",
    # "val_days_6",
    # "val_days_7",
    # "holdout/train",
    # "holdout/test",
    # "new_train",
    "new_test",
    # "new_val",
    "holdout_new_train",
    "holdout_new_test",
    "cherry_train",
    "cherry_val",
    "last_test"
]
#---------------------------------------------------
#                      NCV
#---------------------------------------------------
# Declaring IDs for nested cross validation purposes
TRAIN_IDS = [
    "train_days_1",
    "train_days_12",
    "train_days_123",
    "train_days_1234",
    "train_days_12345",
    "train_days_123456",
]

# They're validation, but in order to keep coherence
# with optimization class they're named test
TEST_IDS = [
    "val_days_2",
    "val_days_3",
    "val_days_4",
    "val_days_5",
    "val_days_6",
    "val_days_7"
]
#---------------------------------------------------

def populate_features():
    result = {}
    for dataset_id in DATASET_IDS:
        # RAW
        result[("raw_feature_tweet_text_token", dataset_id)] = RawFeatureTweetTextToken(dataset_id)
        result[("raw_feature_tweet_hashtags", dataset_id)] = RawFeatureTweetHashtags(dataset_id)
        result[("raw_feature_tweet_id", dataset_id)] = RawFeatureTweetId(dataset_id)
        result[("raw_feature_tweet_media", dataset_id)] = RawFeatureTweetMedia(dataset_id)
        result[("raw_feature_tweet_links", dataset_id)] = RawFeatureTweetLinks(dataset_id)
        result[("raw_feature_tweet_domains", dataset_id)] = RawFeatureTweetDomains(dataset_id)
        result[("raw_feature_tweet_type", dataset_id)] = RawFeatureTweetType(dataset_id)
        result[("raw_feature_tweet_language", dataset_id)] = RawFeatureTweetLanguage(dataset_id)
        result[("raw_feature_tweet_timestamp", dataset_id)] = RawFeatureTweetTimestamp(dataset_id)
        result[("raw_feature_creator_id", dataset_id)] = RawFeatureCreatorId(dataset_id)
        result[("raw_feature_creator_follower_count", dataset_id)] = RawFeatureCreatorFollowerCount(dataset_id)
        result[("raw_feature_creator_following_count", dataset_id)] = RawFeatureCreatorFollowingCount(dataset_id)
        result[("raw_feature_creator_is_verified", dataset_id)] = RawFeatureCreatorIsVerified(dataset_id)
        result[("raw_feature_creator_creation_timestamp", dataset_id)] = RawFeatureCreatorCreationTimestamp(dataset_id)
        result[("raw_feature_engager_id", dataset_id)] = RawFeatureEngagerId(dataset_id)
        result[("raw_feature_engager_follower_count", dataset_id)] = RawFeatureEngagerFollowerCount(dataset_id)
        result[("raw_feature_engager_following_count", dataset_id)] = RawFeatureEngagerFollowingCount(dataset_id)
        result[("raw_feature_engager_is_verified", dataset_id)] = RawFeatureEngagerIsVerified(dataset_id)
        result[("raw_feature_engager_creation_timestamp", dataset_id)] = RawFeatureEngagerCreationTimestamp(dataset_id)
        result[
            ("raw_feature_engagement_creator_follows_engager", dataset_id)] = RawFeatureEngagementCreatorFollowsEngager(
            dataset_id)
        if dataset_id != "test" and dataset_id != "new_test" and dataset_id != "last_test":
            result[("raw_feature_engagement_reply_timestamp", dataset_id)] = RawFeatureEngagementReplyTimestamp(
                dataset_id)
            result[("raw_feature_engagement_retweet_timestamp", dataset_id)] = RawFeatureEngagementRetweetTimestamp(
                dataset_id)
            result[("raw_feature_engagement_comment_timestamp", dataset_id)] = RawFeatureEngagementCommentTimestamp(
                dataset_id)
            result[("raw_feature_engagement_like_timestamp", dataset_id)] = RawFeatureEngagementLikeTimestamp(
                dataset_id)
        # MAPPED
        result[("mapped_feature_tweet_hashtags", dataset_id)] = MappedFeatureTweetHashtags(dataset_id)
        result[("mapped_feature_tweet_id", dataset_id)] = MappedFeatureTweetId(dataset_id)
        result[("mapped_feature_tweet_media", dataset_id)] = MappedFeatureTweetMedia(dataset_id)
        result[("mapped_feature_tweet_links", dataset_id)] = MappedFeatureTweetLinks(dataset_id)
        result[("mapped_feature_tweet_domains", dataset_id)] = MappedFeatureTweetDomains(dataset_id)
        result[("mapped_feature_tweet_language", dataset_id)] = MappedFeatureTweetLanguage(dataset_id)
        result[("mapped_feature_creator_id", dataset_id)] = MappedFeatureCreatorId(dataset_id)
        result[("mapped_feature_engager_id", dataset_id)] = MappedFeatureEngagerId(dataset_id)
        # GENERATED
        # TWEET FEATURE
        # NUMBER OF MEDIA
        result[("tweet_feature_number_of_photo", dataset_id)] = TweetFeatureNumberOfPhoto(dataset_id)
        result[("tweet_feature_number_of_video", dataset_id)] = TweetFeatureNumberOfVideo(dataset_id)
        result[("tweet_feature_number_of_gif", dataset_id)] = TweetFeatureNumberOfGif(dataset_id)
        result[("tweet_feature_number_of_media", dataset_id)] = TweetFeatureNumberOfMedia(dataset_id)
        # HASHTAGS
        result[("tweet_feature_number_of_hashtags", dataset_id)] = TweetFeatureNumberOfHashtags(dataset_id)
        result[("tweet_feature_has_discriminative_hashtag_like", dataset_id)] = HasDiscriminativeHashtag_Like(dataset_id)
        result[("tweet_feature_has_discriminative_hashtag_reply", dataset_id)] = HasDiscriminativeHashtag_Reply(dataset_id)
        result[("tweet_feature_has_discriminative_hashtag_retweet", dataset_id)] = HasDiscriminativeHashtag_Retweet(dataset_id)
        result[("tweet_feature_has_discriminative_hashtag_comment", dataset_id)] = HasDiscriminativeHashtag_Comment(dataset_id)
        result[("tweet_feature_number_of_discriminative_hashtag_like", dataset_id)] = NumberOfDiscriminativeHashtag_Like(dataset_id)
        result[("tweet_feature_number_of_discriminative_hashtag_reply", dataset_id)] = NumberOfDiscriminativeHashtag_Reply(dataset_id)
        result[("tweet_feature_number_of_discriminative_hashtag_retweet", dataset_id)] = NumberOfDiscriminativeHashtag_Retweet(dataset_id)
        result[("tweet_feature_number_of_discriminative_hashtag_comment", dataset_id)] = NumberOfDiscriminativeHashtag_Comment(dataset_id)

        # IS TWEET TYPE
        result[("tweet_feature_is_reply", dataset_id)] = TweetFeatureIsReply(dataset_id)
        result[("tweet_feature_is_retweet", dataset_id)] = TweetFeatureIsRetweet(dataset_id)
        result[("tweet_feature_is_quote", dataset_id)] = TweetFeatureIsQuote(dataset_id)
        result[("tweet_feature_is_top_level", dataset_id)] = TweetFeatureIsTopLevel(dataset_id)
        # IS IN LANGUAGE
        # result[("tweet_is_language_x", dataset_id)] = TweetFeatureIsLanguage(dataset_id, top_popular_language(dataset_id, top_n=10))
        # CREATION TIMESTAMP
        result[("tweet_feature_creation_timestamp_hour", dataset_id)] = TweetFeatureCreationTimestampHour(dataset_id)
        result[("tweet_feature_creation_timestamp_week_day", dataset_id)] = TweetFeatureCreationTimestampWeekDay(dataset_id)
        result[("tweet_feature_creation_timestamp_hour_shifted", dataset_id)] = TweetFeatureCreationTimestampHour_Shifted(dataset_id)
        result[("tweet_feature_creation_timestamp_day_phase", dataset_id)] = TweetFeatureCreationTimestampDayPhase(dataset_id)
        result[("tweet_feature_creation_timestamp_day_phase_shifted", dataset_id)] = TweetFeatureCreationTimestampDayPhase_Shifted(dataset_id)
        # FROM TEXT TOKEN FEATURES
        #result[("tweet_feature_mentions", dataset_id)] = TweetFeatureMappedMentions(dataset_id)
        result[("tweet_feature_number_of_mentions", dataset_id)] = TweetFeatureNumberOfMentions(dataset_id)
        #result[("text_embeddings_clean_PCA_32", dataset_id)] = TweetFeatureTextEmbeddingsPCA32(dataset_id)
        #result[("text_embeddings_clean_PCA_10", dataset_id)] = TweetFeatureTextEmbeddingsPCA10(dataset_id)
        #result[("text_embeddings_hashtags_mentions_LDA_15", dataset_id)] = TweetFeatureTextEmbeddingsHashtagsMentionsLDA15(dataset_id)
        #result[("text_embeddings_hashtags_mentions_LDA_20", dataset_id)] = TweetFeatureTextEmbeddingsHashtagsMentionsLDA20(dataset_id)
        #result[("tweet_feature_dominant_topic_LDA_15", dataset_id)] = TweetFeatureDominantTopicLDA15(dataset_id)
        #result[("tweet_feature_dominant_topic_LDA_20", dataset_id)] = TweetFeatureDominantTopicLDA20(dataset_id)
        result[("tweet_feature_token_length", dataset_id)] = TweetFeatureTokenLength(dataset_id)
        result[("tweet_feature_token_length_unique", dataset_id)] = TweetFeatureTokenLengthUnique(dataset_id)
        result[("tweet_feature_text_token_decoded", dataset_id)] = TweetFeatureTextTokenDecoded(dataset_id)
        result[("tweet_feature_text_topic_word_count_adult_content", dataset_id)] = TweetFeatureTextTopicWordCountAdultContent(dataset_id)
        result[("tweet_feature_text_topic_word_count_kpop", dataset_id)] = TweetFeatureTextTopicWordCountKpop(dataset_id)
        result[("tweet_feature_text_topic_word_count_covid", dataset_id)] = TweetFeatureTextTopicWordCountCovid(dataset_id)
        result[("tweet_feature_text_topic_word_count_sport", dataset_id)] = TweetFeatureTextTopicWordCountSport(dataset_id)

        result[("engager_feature_knows_hashtag_positive", dataset_id)] = EngagerKnowsHashtagPositive(dataset_id)
        result[("engager_feature_knows_hashtag_negative", dataset_id)] = EngagerKnowsHashtagNegative(dataset_id)
        result[("engager_feature_knows_hashtag_like", dataset_id)] = EngagerKnowsHashtagLike(dataset_id)
        result[("engager_feature_knows_hashtag_reply", dataset_id)] = EngagerKnowsHashtagReply(dataset_id)
        result[("engager_feature_knows_hashtag_rt", dataset_id)] = EngagerKnowsHashtagRetweet(dataset_id)
        result[("engager_feature_knows_hashtag_comment", dataset_id)] = EngagerKnowsHashtagComment(dataset_id)
        # DISCRIMINATIVE HASHTAGS
        result[("tweet_feature_has_discriminative_hashtag_like", dataset_id)] = HasDiscriminativeHashtag_Like(dataset_id)
        result[("tweet_feature_has_discriminative_hashtag_reply", dataset_id)] = HasDiscriminativeHashtag_Reply(dataset_id)
        result[("tweet_feature_has_discriminative_hashtag_retweet", dataset_id)] = HasDiscriminativeHashtag_Retweet(dataset_id)
        result[("tweet_feature_has_discriminative_hashtag_comment", dataset_id)] = HasDiscriminativeHashtag_Comment(dataset_id)
        # NUMBER OF PREVIOUS ENGAGEMENTS
        result[("engager_feature_number_of_previous_like_engagement", dataset_id)] = EngagerFeatureNumberOfPreviousLikeEngagement(dataset_id)
        result[("engager_feature_number_of_previous_reply_engagement", dataset_id)] = EngagerFeatureNumberOfPreviousReplyEngagement(dataset_id)
        result[("engager_feature_number_of_previous_retweet_engagement", dataset_id)] = EngagerFeatureNumberOfPreviousRetweetEngagement(dataset_id)
        result[("engager_feature_number_of_previous_comment_engagement", dataset_id)] = EngagerFeatureNumberOfPreviousCommentEngagement(dataset_id)
        result[("engager_feature_number_of_previous_positive_engagement", dataset_id)] = EngagerFeatureNumberOfPreviousPositiveEngagement(dataset_id)
        result[("engager_feature_number_of_previous_negative_engagement", dataset_id)] = EngagerFeatureNumberOfPreviousNegativeEngagement(dataset_id)
        result[("engager_feature_number_of_previous_engagement", dataset_id)] = EngagerFeatureNumberOfPreviousEngagement(dataset_id)
        # NUMBER OF ENGAGEMENTS
        result[("number_of_engagements_like", dataset_id)] = NumberOfEngagementsLike(dataset_id)
        result[("number_of_engagements_retweet", dataset_id)] = NumberOfEngagementsRetweet(dataset_id)
        result[("number_of_engagements_reply", dataset_id)] = NumberOfEngagementsReply(dataset_id)
        result[("number_of_engagements_comment", dataset_id)] = NumberOfEngagementsComment(dataset_id)
        result[("number_of_engagements_negative", dataset_id)] = NumberOfEngagementsNegative(dataset_id)
        result[("number_of_engagements_positive", dataset_id)] = NumberOfEngagementsPositive(dataset_id)
        # NUMBER OF ENGAGEMENTS RATIO
        result[("number_of_engagements_ratio_like", dataset_id)] = NumberOfEngagementsRatioLike(dataset_id)
        result[("number_of_engagements_ratio_retweet", dataset_id)] = NumberOfEngagementsRatioRetweet(dataset_id)
        result[("number_of_engagements_ratio_reply", dataset_id)] = NumberOfEngagementsRatioReply(dataset_id)
        result[("number_of_engagements_ratio_comment", dataset_id)] = NumberOfEngagementsRatioComment(dataset_id)
        result[("number_of_engagements_ratio_negative", dataset_id)] = NumberOfEngagementsRatioNegative(dataset_id)
        result[("number_of_engagements_ratio_positive", dataset_id)] = NumberOfEngagementsRatioPositive(dataset_id)
        # NUMBER OF ENGAGEMENTS WITH LANGUAGE
        result[("number_of_engagements_with_language_like", dataset_id)] = NumberOfEngagementsWithLanguageLike(dataset_id)
        result[("number_of_engagements_with_language_retweet", dataset_id)] = NumberOfEngagementsWithLanguageRetweet(dataset_id)
        result[("number_of_engagements_with_language_reply", dataset_id)] = NumberOfEngagementsWithLanguageReply(dataset_id)
        result[("number_of_engagements_with_language_comment", dataset_id)] = NumberOfEngagementsWithLanguageComment(dataset_id)
        result[("number_of_engagements_with_language_negative", dataset_id)] = NumberOfEngagementsWithLanguageNegative(dataset_id)
        result[("number_of_engagements_with_language_positive", dataset_id)] = NumberOfEngagementsWithLanguagePositive(dataset_id)
        # NUMBER OF ENGAGEMENTS BETWEEN CREATOR AND ENGAGER
        result[("number_of_engagements_between_creator_and_engager_like", dataset_id)] = NumberOfEngagementsBetweenCreatorAndEngagerLike(dataset_id)
        result[("number_of_engagements_between_creator_and_engager_retweet", dataset_id)] = NumberOfEngagementsBetweenCreatorAndEngagerRetweet(dataset_id)
        result[("number_of_engagements_between_creator_and_engager_reply", dataset_id)] = NumberOfEngagementsBetweenCreatorAndEngagerReply(dataset_id)
        result[("number_of_engagements_between_creator_and_engager_comment", dataset_id)] = NumberOfEngagementsBetweenCreatorAndEngagerComment(dataset_id)
        result[("number_of_engagements_between_creator_and_engager_negative", dataset_id)] = NumberOfEngagementsBetweenCreatorAndEngagerNegative(dataset_id)
        result[("number_of_engagements_between_creator_and_engager_positive", dataset_id)] = NumberOfEngagementsBetweenCreatorAndEngagerPositive(dataset_id)
        # NUMBER OF PREVIOUS ENGAGEMENTS WITH LANGUAGE
        result[("engager_feature_number_of_previous_like_engagement_with_language",dataset_id)] = EngagerFeatureNumberOfPreviousLikeEngagementWithLanguage(dataset_id)
        result[("engager_feature_number_of_previous_reply_engagement_with_language",dataset_id)] = EngagerFeatureNumberOfPreviousReplyEngagementWithLanguage(dataset_id)
        result[("engager_feature_number_of_previous_retweet_engagement_with_language",dataset_id)] = EngagerFeatureNumberOfPreviousRetweetEngagementWithLanguage(dataset_id)
        result[("engager_feature_number_of_previous_comment_engagement_with_language",dataset_id)] = EngagerFeatureNumberOfPreviousCommentEngagementWithLanguage(dataset_id)
        result[("engager_feature_number_of_previous_positive_engagement_with_language",dataset_id)] = EngagerFeatureNumberOfPreviousPositiveEngagementWithLanguage(dataset_id)
        result[("engager_feature_number_of_previous_negative_engagement_with_language",dataset_id)] = EngagerFeatureNumberOfPreviousNegativeEngagementWithLanguage(dataset_id)
        # NUMBER OF PREVIOUS ENGAGEMENTS A CREATOR HAS RECEIVED
        result[("creator_feature_number_of_previous_like_engagements_received",dataset_id)] = CreatorNumberOfPreviousEngagementReceivedLike(dataset_id)
        result[("creator_feature_number_of_previous_reply_engagements_received",dataset_id)] = CreatorNumberOfPreviousEngagementReceivedReply(dataset_id)
        result[("creator_feature_number_of_previous_retweet_engagements_received",dataset_id)] = CreatorNumberOfPreviousEngagementReceivedRetweet(dataset_id)
        result[("creator_feature_number_of_previous_comment_engagements_received",dataset_id)] = CreatorNumberOfPreviousEngagementReceivedComment(dataset_id)
        result[("creator_feature_number_of_previous_positive_engagements_received",dataset_id)] = CreatorNumberOfPreviousEngagementReceivedPositive(dataset_id)
        result[("creator_feature_number_of_previous_negative_engagements_received",dataset_id)] = CreatorNumberOfPreviousEngagementReceivedNegative(dataset_id)
        # NUMBER OF PREVIOUS ENGAGEMENTS A CREATOR HAS GIVEN
        result[("creator_feature_number_of_previous_like_engagements_given",dataset_id)] = CreatorNumberOfPreviousEngagementGivenLike(dataset_id)
        result[("creator_feature_number_of_previous_reply_engagements_given",dataset_id)] = CreatorNumberOfPreviousEngagementGivenReply(dataset_id)
        result[("creator_feature_number_of_previous_retweet_engagements_given",dataset_id)] = CreatorNumberOfPreviousEngagementGivenRetweet(dataset_id)
        result[("creator_feature_number_of_previous_comment_engagements_given",dataset_id)] = CreatorNumberOfPreviousEngagementGivenComment(dataset_id)
        result[("creator_feature_number_of_previous_positive_engagements_given",dataset_id)] = CreatorNumberOfPreviousEngagementGivenPositive(dataset_id)
        result[("creator_feature_number_of_previous_negative_engagements_given",dataset_id)] = CreatorNumberOfPreviousEngagementGivenNegative(dataset_id)
        # NUMBER OF ENGAGEMENTS A CREATOR HAS RECEIVED
        result[("creator_feature_number_of_like_engagements_received",dataset_id)] = CreatorNumberOfEngagementReceivedLike(dataset_id)
        result[("creator_feature_number_of_reply_engagements_received",dataset_id)] = CreatorNumberOfEngagementReceivedReply(dataset_id)
        result[("creator_feature_number_of_retweet_engagements_received",dataset_id)] = CreatorNumberOfEngagementReceivedRetweet(dataset_id)
        result[("creator_feature_number_of_comment_engagements_received",dataset_id)] = CreatorNumberOfEngagementReceivedComment(dataset_id)
        result[("creator_feature_number_of_positive_engagements_received",dataset_id)] = CreatorNumberOfEngagementReceivedPositive(dataset_id)
        result[("creator_feature_number_of_negative_engagements_received",dataset_id)] = CreatorNumberOfEngagementReceivedNegative(dataset_id)
        # NUMBER OF PREVIOUS ENGAGEMENTS A CREATOR HAS GIVEN
        result[("creator_feature_number_of_like_engagements_given",dataset_id)] = CreatorNumberOfEngagementGivenLike(dataset_id)
        result[("creator_feature_number_of_reply_engagements_given",dataset_id)] = CreatorNumberOfEngagementGivenReply(dataset_id)
        result[("creator_feature_number_of_retweet_engagements_given",dataset_id)] = CreatorNumberOfEngagementGivenRetweet(dataset_id)
        result[("creator_feature_number_of_comment_engagements_given",dataset_id)] = CreatorNumberOfEngagementGivenComment(dataset_id)
        result[("creator_feature_number_of_positive_engagements_given",dataset_id)] = CreatorNumberOfEngagementGivenPositive(dataset_id)
        result[("creator_feature_number_of_negative_engagements_given",dataset_id)] = CreatorNumberOfEngagementGivenNegative(dataset_id)
        # NUMBER OF ENGAGEMENTS AN ENGAGER HAS RECEIVED
        result[("engager_feature_number_of_like_engagements_received",dataset_id)] = EngagerNumberOfEngagementReceivedLike(dataset_id)
        result[("engager_feature_number_of_reply_engagements_received",dataset_id)] = EngagerNumberOfEngagementReceivedReply(dataset_id)
        result[("engager_feature_number_of_retweet_engagements_received",dataset_id)] = EngagerNumberOfEngagementReceivedRetweet(dataset_id)
        result[("engager_feature_number_of_comment_engagements_received",dataset_id)] = EngagerNumberOfEngagementReceivedComment(dataset_id)
        result[("engager_feature_number_of_positive_engagements_received",dataset_id)] = EngagerNumberOfEngagementReceivedPositive(dataset_id)
        result[("engager_feature_number_of_negative_engagements_received",dataset_id)] = EngagerNumberOfEngagementReceivedNegative(dataset_id)
        # NUMBER OF ENGAGEMENTS A TWEET HAS RECEIVED
        # result[("tweet_feature_number_of_like_engagements", dataset_id)] = TweetNumberOfEngagementLike(dataset_id)
        # result[("tweet_feature_number_of_reply_engagements", dataset_id)] = TweetNumberOfEngagementReply(dataset_id)
        # result[("tweet_feature_number_of_retweet_engagements", dataset_id)] = TweetNumberOfEngagementRetweet(dataset_id)
        # result[("tweet_feature_number_of_comment_engagements", dataset_id)] = TweetNumberOfEngagementComment(dataset_id)
        # result[("tweet_feature_number_of_positive_engagements", dataset_id)] = TweetNumberOfEngagementPositive(dataset_id)
        # result[("tweet_feature_number_of_negative_engagements", dataset_id)] = TweetNumberOfEngagementNegative(dataset_id)
        # NUMBER OF PREVIOUS ENGAGEMENTS A TWEET HAS RECEIVED
        # result[("tweet_feature_number_of_previous_like_engagements",dataset_id)] = TweetNumberOfPreviousEngagementLike(dataset_id)
        # result[("tweet_feature_number_of_previous_reply_engagements",dataset_id)] = TweetNumberOfPreviousEngagementReply(dataset_id)
        # result[("tweet_feature_number_of_previous_retweet_engagements",dataset_id)] = TweetNumberOfPreviousEngagementRetweet(dataset_id)
        # result[("tweet_feature_number_of_previous_comment_engagements",dataset_id)] = TweetNumberOfPreviousEngagementComment(dataset_id)
        # result[("tweet_feature_number_of_previous_positive_engagements",dataset_id)] = TweetNumberOfPreviousEngagementPositive(dataset_id)
        # result[("tweet_feature_number_of_previous_negative_engagements",dataset_id)] = TweetNumberOfPreviousEngagementNegative(dataset_id)
        # NUMBER OF PREVIOUS ENGAGEMENTS RATIO
        result[("engager_feature_number_of_previous_like_engagement_ratio", dataset_id)] = EngagerFeatureNumberOfPreviousLikeEngagementRatio(dataset_id)
        result[("engager_feature_number_of_previous_reply_engagement_ratio", dataset_id)] = EngagerFeatureNumberOfPreviousReplyEngagementRatio(dataset_id)
        result[("engager_feature_number_of_previous_retweet_engagement_ratio", dataset_id)] = EngagerFeatureNumberOfPreviousRetweetEngagementRatio(dataset_id)
        result[("engager_feature_number_of_previous_comment_engagement_ratio", dataset_id)] = EngagerFeatureNumberOfPreviousCommentEngagementRatio(dataset_id)
        result[("engager_feature_number_of_previous_positive_engagement_ratio", dataset_id)] = EngagerFeatureNumberOfPreviousPositiveEngagementRatio(dataset_id)
        result[("engager_feature_number_of_previous_negative_engagement_ratio", dataset_id)] = EngagerFeatureNumberOfPreviousNegativeEngagementRatio(dataset_id)
        # NUMBER OF PREVIOUS ENGAGEMENTS RATIO WITH -1 FOR COLD USERS
        result[("engager_feature_number_of_previous_like_engagement_ratio_1", dataset_id)] = EngagerFeatureNumberOfPreviousLikeEngagementRatio1(dataset_id)
        result[("engager_feature_number_of_previous_reply_engagement_ratio_1", dataset_id)] = EngagerFeatureNumberOfPreviousReplyEngagementRatio1(dataset_id)
        result[("engager_feature_number_of_previous_retweet_engagement_ratio_1", dataset_id)] = EngagerFeatureNumberOfPreviousRetweetEngagementRatio1(dataset_id)
        result[("engager_feature_number_of_previous_comment_engagement_ratio_1", dataset_id)] = EngagerFeatureNumberOfPreviousCommentEngagementRatio1(dataset_id)
        result[("engager_feature_number_of_previous_positive_engagement_ratio_1", dataset_id)] = EngagerFeatureNumberOfPreviousPositiveEngagementRatio1(dataset_id)
        result[("engager_feature_number_of_previous_negative_engagement_ratio_1", dataset_id)] = EngagerFeatureNumberOfPreviousNegativeEngagementRatio1(dataset_id)
        # NUMBER OF PREVIOUS ENGAGEMENTS BETWEEN CREATOR AND ENGAGER BY CREATOR
        result[("engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_creator", dataset_id)] = EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByCreator(dataset_id)
        result[("engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_creator", dataset_id)] = EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByCreator(dataset_id)
        result[("engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_creator", dataset_id)] = EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByCreator(dataset_id)
        result[("engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_creator", dataset_id)] = EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByCreator(dataset_id)
        result[("engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_creator", dataset_id)] = EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByCreator(dataset_id)
        result[("engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_creator", dataset_id)] = EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByCreator(dataset_id)
        # NUMBER OF PREVIOUS ENGAGEMENTS BETWEEN CREATOR AND ENGAGER BY ENGAGER
        result[("engager_feature_number_of_previous_like_engagement_between_creator_and_engager_by_engager", dataset_id)] = EngagerFeatureNumberOfPreviousLikeEngagementBetweenCreatorAndEngagerByEngager(dataset_id)
        result[("engager_feature_number_of_previous_reply_engagement_between_creator_and_engager_by_engager", dataset_id)] = EngagerFeatureNumberOfPreviousReplyEngagementBetweenCreatorAndEngagerByEngager(dataset_id)
        result[("engager_feature_number_of_previous_retweet_engagement_between_creator_and_engager_by_engager", dataset_id)] = EngagerFeatureNumberOfPreviousRetweetEngagementBetweenCreatorAndEngagerByEngager(dataset_id)
        result[("engager_feature_number_of_previous_comment_engagement_between_creator_and_engager_by_engager", dataset_id)] = EngagerFeatureNumberOfPreviousCommentEngagementBetweenCreatorAndEngagerByEngager(dataset_id)
        result[("engager_feature_number_of_previous_negative_engagement_between_creator_and_engager_by_engager", dataset_id)] = EngagerFeatureNumberOfPreviousNegativeEngagementBetweenCreatorAndEngagerByEngager(dataset_id)
        result[("engager_feature_number_of_previous_positive_engagement_between_creator_and_engager_by_engager", dataset_id)] = EngagerFeatureNumberOfPreviousPositiveEngagementBetweenCreatorAndEngagerByEngager(dataset_id)
        # MAIN LANGUAGE
        result[("engager_main_language", dataset_id)] = EngagerMainLanguage(dataset_id)
        result[("creator_main_language", dataset_id)] = CreatorMainLanguage(dataset_id)
        result[("creator_and_engager_have_same_main_language", dataset_id)] = CreatorAndEngagerHaveSameMainLanguage(dataset_id)
        result[("is_tweet_in_creator_main_language", dataset_id)] = IsTweetInCreatorMainLanguage(dataset_id)
        result[("is_tweet_in_engager_main_language", dataset_id)] = IsTweetInEngagerMainLanguage(dataset_id)
        #result[("statistical_probability_main_language_of_engager_engage_tweet_language_1", dataset_id)] = StatisticalProbabilityMainLanguageOfEngagerEngageTweetLanguage1(dataset_id)
        #result[("statistical_probability_main_language_of_engager_engage_tweet_language_2", dataset_id)] = StatisticalProbabilityMainLanguageOfEngagerEngageTweetLanguage2(dataset_id)
        # MAIN LANGUAGE
        result[("engager_main_grouped_language", dataset_id)] = EngagerMainGroupedLanguage(dataset_id)
        result[("creator_main_grouped_language", dataset_id)] = CreatorMainGroupedLanguage(dataset_id)
        result[("creator_and_engager_have_same_main_grouped_language", dataset_id)] = CreatorAndEngagerHaveSameMainGroupedLanguage(dataset_id)
        result[("is_tweet_in_creator_main_grouped_language", dataset_id)] = IsTweetInCreatorMainGroupedLanguage(dataset_id)
        result[("is_tweet_in_engager_main_grouped_language", dataset_id)] = IsTweetInEngagerMainGroupedLanguage(dataset_id)
        # FOLD ENSEMBLING
        #result[("xgb_fold_ensembling_like_1", dataset_id)] = XGBFoldEnsemblingLike1(dataset_id)
        #result[("xgb_fold_ensembling_retweet_1", dataset_id)] = XGBFoldEnsemblingRetweet1(dataset_id)
        #result[("xgb_fold_ensembling_reply_1", dataset_id)] = XGBFoldEnsemblingReply1(dataset_id)
        #result[("xgb_fold_ensembling_comment_1", dataset_id)] = XGBFoldEnsemblingComment1(dataset_id)
        result[("xgb_fold_ensembling_like_2", dataset_id)] = XGBFoldEnsemblingLike2(dataset_id)
        result[("xgb_fold_ensembling_retweet_2", dataset_id)] = XGBFoldEnsemblingRetweet2(dataset_id)
        result[("xgb_fold_ensembling_reply_2", dataset_id)] = XGBFoldEnsemblingReply2(dataset_id)
        result[("xgb_fold_ensembling_comment_2", dataset_id)] = XGBFoldEnsemblingComment2(dataset_id)


        result[("lgbm_fold_ensembling_like_1", dataset_id)] = LGBMFoldEnsemblingLike1(dataset_id)
        # SIMILARITY FOLD ENSEMBLING
        #result[("hashtag_similarity_fold_ensembling_positive", dataset_id)] = HashtagSimilarityFoldEnsembling(dataset_id, label="positive")
        #result[("link_similarity_fold_ensembling_positive", dataset_id)] = HashtagSimilarityFoldEnsembling(dataset_id, label="positive")
        #result[("domain_similarity_fold_ensembling_positive", dataset_id)] = HashtagSimilarityFoldEnsembling(dataset_id, label="positive")

        # GRAPH
        result[("graph_two_steps_positive", dataset_id)] = GraphTwoStepsPositive(dataset_id)
        result[("graph_two_steps_negative", dataset_id)] = GraphTwoStepsNegative(dataset_id)
        result[("graph_two_steps_like", dataset_id)] = GraphTwoStepsLike(dataset_id)
        result[("graph_two_steps_reply", dataset_id)] = GraphTwoStepsReply(dataset_id)
        result[("graph_two_steps_retweet", dataset_id)] = GraphTwoStepsRetweet(dataset_id)
        result[("graph_two_steps_comment", dataset_id)] = GraphTwoStepsComment(dataset_id)

        # ADJACENCY
        result[("graph_two_steps_adjacency_positive", dataset_id)] = GraphTwoStepsAdjacencyPositive(dataset_id)
        result[("graph_two_steps_adjacency_negative", dataset_id)] = GraphTwoStepsAdjacencyNegative(dataset_id)
        result[("graph_two_steps_adjacency_like", dataset_id)] = GraphTwoStepsAdjacencyLike(dataset_id)
        result[("graph_two_steps_adjacency_reply", dataset_id)] = GraphTwoStepsAdjacencyReply(dataset_id)
        result[("graph_two_steps_adjacency_retweet", dataset_id)] = GraphTwoStepsAdjacencyRetweet(dataset_id)
        result[("graph_two_steps_adjacency_comment", dataset_id)] = GraphTwoStepsAdjacencyComment(dataset_id)

        result[("adjacency_between_creator_and_engager_positive", dataset_id)] = AdjacencyBetweenCreatorAndEngagerPositive(dataset_id)
        result[("adjacency_between_creator_and_engager_negative", dataset_id)] = AdjacencyBetweenCreatorAndEngagerNegative(dataset_id)
        result[("adjacency_between_creator_and_engager_like", dataset_id)] = AdjacencyBetweenCreatorAndEngagerLike(dataset_id)
        result[("adjacency_between_creator_and_engager_reply", dataset_id)] = AdjacencyBetweenCreatorAndEngagerReply(dataset_id)
        result[("adjacency_between_creator_and_engager_retweet", dataset_id)] = AdjacencyBetweenCreatorAndEngagerRetweet(dataset_id)
        result[("adjacency_between_creator_and_engager_comment", dataset_id)] = AdjacencyBetweenCreatorAndEngagerComment(dataset_id)

        # IS ENGAGEMENT TYPE
        if dataset_id != "test" and dataset_id != "new_test" and dataset_id != "last_test":
            result[("tweet_feature_engagement_is_like", dataset_id)] = TweetFeatureEngagementIsLike(dataset_id)
            result[("tweet_feature_engagement_is_retweet", dataset_id)] = TweetFeatureEngagementIsRetweet(dataset_id)
            result[("tweet_feature_engagement_is_comment", dataset_id)] = TweetFeatureEngagementIsComment(dataset_id)
            result[("tweet_feature_engagement_is_reply", dataset_id)] = TweetFeatureEngagementIsReply(dataset_id)
            result[("tweet_feature_engagement_is_positive", dataset_id)] = TweetFeatureEngagementIsPositive(dataset_id)
            result[("tweet_feature_engagement_is_negative", dataset_id)] = TweetFeatureEngagementIsNegative(dataset_id)

        # HASHTAG POPULARITY
        # SET 1
        result[("max_hashtag_popularity_1", dataset_id)] = MaxHashtagPopularity(dataset_id, 1000000, 950000)
        result[("min_hashtag_popularity_1", dataset_id)] = MinHashtagPopularity(dataset_id, 1000000, 950000)
        result[("mean_hashtag_popularity_1", dataset_id)] = MeanHashtagPopularity(dataset_id, 1000000, 950000)
        result[("total_hashtag_popularity_1", dataset_id)] = TotalHashtagPopularity(dataset_id, 1000000, 950000)
        # SET 2
        result[("max_hashtag_popularity_2", dataset_id)] = MaxHashtagPopularity(dataset_id, 2000000, 950000)
        result[("min_hashtag_popularity_2", dataset_id)] = MinHashtagPopularity(dataset_id, 2000000, 950000)
        result[("mean_hashtag_popularity_2", dataset_id)] = MeanHashtagPopularity(dataset_id, 2000000, 950000)
        result[("total_hashtag_popularity_2", dataset_id)] = TotalHashtagPopularity(dataset_id, 2000000, 950000)
        # SET 3
        result[("max_hashtag_popularity_3", dataset_id)] = MaxHashtagPopularity(dataset_id, 100000, 99000)
        result[("min_hashtag_popularity_3", dataset_id)] = MinHashtagPopularity(dataset_id, 100000, 99000)
        result[("mean_hashtag_popularity_3", dataset_id)] = MeanHashtagPopularity(dataset_id, 100000, 99000)
        result[("total_hashtag_popularity_3", dataset_id)] = TotalHashtagPopularity(dataset_id, 100000, 99000)

        # CREATOR FEATURE
        #result[("creator_feature_frequency_of_unique_tokens", dataset_id)] = CreatorFrequencyUniqueTokens(dataset_id)
        # KNOWN COUNT OF ENGAGEMENT
        # BAD IMPLEMENTATION - DOES NOT RESPECT TIME
        # result[("engager_feature_known_number_of_like_engagement", dataset_id)] = EngagerFeatureKnowNumberOfLikeEngagement(dataset_id)
        # result[("engager_feature_known_number_of_reply_engagement", dataset_id)] = EngagerFeatureKnowNumberOfReplyEngagement(dataset_id)
        # result[("engager_feature_known_number_of_retweet_engagement", dataset_id)] = EngagerFeatureKnowNumberOfRetweetEngagement(dataset_id)
        # result[("engager_feature_known_number_of_comment_engagement", dataset_id)] = EngagerFeatureKnowNumberOfCommentEngagement(dataset_id)
        # result[("engager_feature_known_number_of_positive_engagement", dataset_id)] = EngagerFeatureKnowNumberOfPositiveEngagement(dataset_id)
        # result[("engager_feature_known_number_of_negative_engagement", dataset_id)] = EngagerFeatureKnowNumberOfNegativeEngagement(dataset_id)
        # KNOW TWEET LANGUAGE
        # BAD IMPLEMENTATION - DOES NOT RESPECT TIME
        # result[("engager_feature_know_tweet_language", dataset_id)] = EngagerFeatureKnowTweetLanguage(dataset_id)

    return result


FEATURES = populate_features()

DICTIONARIES = {
    "mapping_tweet_id_direct": MappingTweetIdDictionary(inverse=False),
    "mapping_tweet_id_inverse": MappingTweetIdDictionary(inverse=True),
    "mapping_user_id_direct": MappingUserIdDictionary(inverse=False),
    "mapping_user_id_inverse": MappingUserIdDictionary(inverse=True),
    "mapping_language_id_direct": MappingLanguageDictionary(inverse=False),
    "mapping_language_id_inverse": MappingLanguageDictionary(inverse=True),
    "mapping_domain_id_direct": MappingDomainDictionary(inverse=False),
    "mapping_domain_id_inverse": MappingDomainDictionary(inverse=True),
    "mapping_link_id_direct": MappingLinkDictionary(inverse=False),
    "mapping_link_id_inverse": MappingLinkDictionary(inverse=True),
    "mapping_media_id_direct": MappingMediaDictionary(inverse=False),
    "mapping_media_id_inverse": MappingMediaDictionary(inverse=True),
    "mapping_hashtag_id_direct": MappingHashtagDictionary(inverse=False),
    "mapping_hashtag_id_inverse": MappingHashtagDictionary(inverse=True)
}

DICT_ARRAYS = {
    # TWEET BASIC FEATURES
    "hashtags_tweet_dict_array": HashtagsTweetBasicFeatureDictArray(),
    "media_tweet_dict_array": MediaTweetBasicFeatureDictArray(),
    "links_tweet_dict_array": LinksTweetBasicFeatureDictArray(),
    "domains_tweet_dict_array": DomainsTweetBasicFeatureDictArray(),
    "type_tweet_dict_array": TypeTweetBasicFeatureDictArray(),
    "timestamp_tweet_dict_array": TimestampTweetBasicFeatureDictArray(),
    "creator_id_tweet_dict_array": CreatorIdTweetBasicFeatureDictArray(),
    # USER BASIC FEATURES
    "follower_count_user_dict_array": FollowerCountUserBasicFeatureDictArray(),
    "following_count_user_dict_array": FollowingCountUserBasicFeatureDictArray(),
    "is_verified_user_dict_array": IsVerifiedUserBasicFeatureDictArray(),
    "creation_timestamp_user_dict_array": CreationTimestampUserBasicFeatureDictArray(),
    "language_user_dict_array": LanguageUserBasicFeatureDictArray(),
    # TWEET TEXT FEATURE
    #"text_embeddings_PCA_10_feature_dict_array": TweetTextEmbeddingsPCA10FeatureDictArray(),
    #"text_embeddings_PCA_32_feature_dict_array": TweetTextEmbeddingsPCA32FeatureDictArray(),
    "mapping_mentions_id_dictionary": MappingMentionsDictionary(),
    #"text_embeddings_hashtags_mentions_LDA_15_feature_dict_array": TweetTextEmbeddingsHashtagsMentionsLDA15FeatureDictArray(),
    #"text_embeddings_hashtags_mentions_LDA_20_feature_dict_array": TweetTextEmbeddingsHashtagsMentionsLDA20FeatureDictArray()
    "tweet_token_length_feature_dict_array": TweetTokenLengthFeatureDictArray(),
    "tweet_token_length_unique_feature_dict_array": TweetTokenLengthUniqueFeatureDictArray(),

}

SPARSE_MATRIXES = {
    # ICM
    "tweet_hashtags_csr_matrix": HashtagMatrix(),
    "tweet_links_csr_matrix": LinkMatrix(),
    "tweet_domains_csr_matrix": DomainMatrix(),
    "tweet_language_csr_matrix": LanguageMatrixOnlyPositive()
}


def create_all():
    # For more parallelism
    features_grouped = [[v for k, v in FEATURES.items() if k[1] == dataset_id] for dataset_id in DATASET_IDS]
    with mp.Pool(8) as pool:
        pool.map(create_features, features_grouped)
    # list(map(create_feature, FEATURES.values()))
    list(map(create_dictionary, DICTIONARIES.values()))
    list(map(create_dictionary, DICT_ARRAYS.values()))
    list(map(create_matrix, SPARSE_MATRIXES.values()))


def create_features(feature_list: list):
    list(map(create_feature, feature_list))


def create_feature(feature: Feature):
    if not feature.has_feature():
        print(f"creating: {feature.dataset_id}_{feature.feature_name}")
        feature.create_feature()
    else:
        print(f"already created: {feature.dataset_id}_{feature.feature_name}")


def create_dictionary(dictionary: Dictionary):
    if not dictionary.has_dictionary():
        print(f"creating: {dictionary.dictionary_name}")
        dictionary.create_dictionary()
    else:
        print(f"already created: {dictionary.dictionary_name}")


def create_matrix(matrix: CSR_SparseMatrix):
    if not matrix.has_matrix():
        print(f"creating: {matrix.matrix_name}")
        matrix.create_matrix()
    else:
        print(f"already created: {matrix.matrix_name}")


def consistency_check(dataset_id: str):
    features = np.array(FEATURES.items())
    lenghts = np.array([len(v.load_or_create()) for k, v in FEATURES.items() if k[1] == dataset_id])
    if all(lenghts == lenghts[0]):
        print(f"{dataset_id} is consistent")
    else:
        not_consistent_features_mask = lenghts != lenghts[0]
        for feature in features[lenghts][not_consistent_features_mask]:
            print(feature)


def consistency_check_all():
    for dataset_id in DATASET_IDS:
        consistency_check(dataset_id)

def to_svm(arg, filename):
    i = arg[0]
    data = arg[1]
    skd.dump_svmlight_file(
        X=data[0],
        y=data[1][data[1].columns[0]].array,
        f=f"temp/{i}_{filename}.svm"
    )

def cache_dataset_as_svm(filename, X_train, Y_train=None, no_fuck_my_self=False):
    if Y_train is None:
        Y_train = pd.DataFrame(np.full(len(X_train), 0))
    if pathlib.Path(f"{filename}.svm").exists():
        print("file already exists, be careful to overwrite it")
    else:
        # if no_fuck_my_self:
        #     skd.dump_svmlight_file(
        #         X=X_train,
        #         y=Y_train[Y_train.columns[0]].array,
        #         f=f"{filename}.svm"
        #     )
        # else:
        
        X_chunks = np.array_split(X_train, 1000)
        Y_chunks = np.array_split(Y_train, 1000)

        pathlib.Path("temp").mkdir(parents=True, exist_ok=True)

        partial_to_svm = functools.partial(to_svm, filename=filename)
        with mp.Pool(multiprocessing.cpu_count()) as p:
            p.map(partial_to_svm, enumerate(zip(X_chunks, Y_chunks)))

        cmd = f'cat {"".join([f"temp/{i}_{filename}.svm " for i in range(1000)])}> {filename}.svm'
        os.system(cmd)
        cmd = f'rm -r temp'
        os.system(cmd)

