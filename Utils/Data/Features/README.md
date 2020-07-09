# Example
### Get feature
```python
from Utils.Data.Data import get_feature
feature_df = get_feature(feature_name="mapped_feature_tweet_id", dataset_id="train")
```

### Get multiple feature
```python
from Utils.Data.Data import get_dataset
features = [
    "tweet_feature_number_of_photo",
    "tweet_feature_number_of_media",
    "tweet_feature_number_of_mentions"
]
feature_df = get_dataset(features=features, dataset_id="train")
```

# Features

For each dataset we have the following features:

### Raw Features

- **raw_feature_tweet_text_token**: str:
 <br>Ordered list of Bert ids corresponding to Bert tokenization of Tweet text.
- **raw_feature_tweet_hashtags**: str:
 <br>Tab separated list of hastags (hashed identifiers) present in the tweet.
- **raw_feature_tweet_text_token**: str:
 <br>Tweet identifier (hashed).
- **raw_feature_tweet_media**: str:
 <br>Tab separated list of media types. Media type can be in (Photo, Video, Gif)
- **raw_feature_tweet_links**: str:
 <br>Tab separeted list of links (hashed identifiers) included in the Tweet.
- **raw_feature_tweet_domains**: str:
 <br>Tab separated list of hashed domains included in the Tweet (twitter.com, dogs.com).
- **raw_feature_tweet_type**: str:
 <br>Tweet type, can be either Retweet, Quote, Reply, or Toplevel.
- **raw_feature_tweet_language**: str:
 <br>Identifier corresponding to the inferred language of the Tweet.
- **raw_feature_tweet_timestamp**: int:
 <br>Unix timestamp, in sec of the creation time of the Tweet.
 

- **raw_feature_creator_id**: str:
 <br>User identifier.
- **raw_feature_creator_follower_count**: int:
 <br>Number of followers of the user.
- **raw_feature_creator_following_count**: int:
 <br>Number of accounts the user is following.
- **raw_feature_creator_is_verified**: bool:
 <br>Is the account verified?
- **raw_feature_creator_creation_timestamp**: int:
 <br>Unix timestamp, in seconds, of the creation time of the account.
 
 
 - **raw_feature_creator_id**: str:
 <br>User identifier.
- **raw_feature_creator_follower_count**: int:
 <br>Number of followers of the user.
- **raw_feature_creator_following_count**: int:
 <br>Number of accounts the user is following.
- **raw_feature_creator_is_verified**: bool:
 <br>Is the account verified?
- **raw_feature_creator_creation_timestamp**: int:
 <br>Unix timestamp, in seconds, of the creation time of the account.
 
 
- **raw_feature_engagement_creator_follows_engager**: bool:
 <br>Does the account of the engaged tweet author follow the account that has made the engagement?
- **raw_feature_engagement_reply_timestamp**: int: **[only train/val set]**
 <br>If there is at least one, unix timestamp, in s, of one of the replies.
- **raw_feature_engagement_retweet_timestamp**: int: **[only train/val set]**
 <br>If there is one, unix timestamp, in s, of the retweet of the tweet by the engaging user.
- **raw_feature_engagement_comment_timestamp**: int: **[only train/val set]**
 <br>If there is at least one, unix timestamp, in s, of one of the retweet with comment of the tweet by the engaging user.
- **raw_feature_engagement_like_timestamp**: int: **[only train/val set]**
 <br>If there is one, Unix timestamp, in s, of the like.
 
 ### Mapped Features
 
 These features are just the same as raw features but each identifier has been mapped to a positive integer:
 
 - **mapped_feature_tweet_hashtags**: list of int:
 <br>List of hashtags present in the tweet. *None* otherwise.
  - **mapped_feature_tweet_id**: int:
 <br>Tweet identifier.
  - **mapped_feature_tweet_media**: list of int:
 <br>List of media present in the tweet. *None* otherwise.
  - **mapped_feature_tweet_links**: list of int:
 <br>List of links present in the tweet. *None* otherwise.
  - **mapped_feature_tweet_domains**: list of int:
 <br>List of domains present in the tweet. *None* otherwise.
  - **mapped_feature_tweet_language**: int:
 <br>Tweet language.
  - **mapped_feature_creator_id**: int:
 <br>User identifier of the creator.
  - **mapped_feature_engager_id**: int:
 <br>User identifier of the engager.
 
 ### Generated Features
 
 These features has been extracted from the previous ones. All the identifiers used in these features are mapped using the internal dictionary.

 ### Generated Tweet Features
 
 #### Number of media

  - **tweet_feature_number_of_photo**: int:
 <br>Number of photo in the tweet.
  - **tweet_feature_number_of_video**: int:
 <br>Number of video in the tweet.
  - **tweet_feature_number_of_gif**: int:
 <br>Number of gif in the tweet.
   - **tweet_feature_number_of_media**: int:
 <br>Number of media (photo, video and gif) in the tweet.
 
 #### Number of hashtags
 
   - **tweet_feature_number_of_hashtags**: int:
 <br>Number of hashtags in the tweet.
 
 #### Hashtag related features
 
   - **tweet_feature_has_discriminative_hashtag_like**: bool:
 <br>If the tweet contains a discriminative (w.r.t like engagements) hashtag.
   - **tweet_feature_has_discriminative_hashtag_retweet**: bool:
 <br>If the tweet contains a discriminative (w.r.t retweet engagements) hashtag.
   - **tweet_feature_has_discriminative_hashtag_reply**: bool:
 <br>If the tweet contains a discriminative (w.r.t reply engagements) hashtag.
   - **tweet_feature_has_discriminative_hashtag_comment**: bool:
 <br>If the tweet contains a discriminative (w.r.t comment engagements) hashtag.
 
   - **tweet_feature_number_of_discriminative_hashtag_like**: int:
 <br>Count of the discriminative (w.r.t like engagements) hashtags contained in a tweet.
   - **tweet_feature_number_of_discriminative_hashtag_retweet**: int:
 <br>Count of the discriminative (w.r.t retweet engagements) hashtags contained in a tweet.
   - **tweet_feature_number_of_discriminative_hashtag_reply**: int:
 <br>Count of the discriminative (w.r.t reply engagements) hashtags contained in a tweet.
   - **tweet_feature_number_of_discriminative_hashtag_comment**: int:
 <br>Count of the discriminative (w.r.t comment engagements) hashtags contained in a tweet.
 
 
 #### Is tweet type

  - **tweet_feature_is_reply**: bool:
 <br>True if the tweet is a reply.
  - **tweet_feature_is_retweet**: bool:
 <br>True if the tweet is a retweet.
  - **tweet_feature_is_quote**: bool:
 <br>True if the tweet is a quote.
   - **tweet_feature_is_top_level**: bool:
 <br>True if the tweet is a top_level.
 
 #### Extracted from text token
 
   - **tweet_feature_mentions**: list of ints (or None):
 <br>Mentions extracted from the tweet. 
   - **tweet_feature_number_of_mentions**: int:
 <br>Number of mentions in the tweet.
 
   - **tweet_feature_token_length**: int:
 <br>Number of BERT tokens in the tweet.
   - **tweet_feature_token_length_unique**: int:
 <br>Number of unique bert tokens in the tweet.
 
   - **tweet_feature_text_token_decoded**: list of str:
 <br>Decoded BERT tokens.
 
   - **tweet_feature_text_topic_word_count_adult_content**: int:
 <br>Number of 'adult content' words.
   - **tweet_feature_text_topic_word_count_kpop**: int:
 <br>Number of 'kpop' words.
   - **tweet_feature_text_topic_word_count_covid**: int:
 <br>Number of 'covid' words.
   - **tweet_feature_text_topic_word_count_sport**: int:
 <br>Number of 'sport' words.
   
 
 #### Creation timestamp
 
   - **tweet_feature_creation_timestamp_hour**: int:
 <br>The hour when the tweet has been created. (0-23 UTC hour)
   - **tweet_feature_creation_timestamp_hour_shifted**: int:
 <br>The shifted hour (+12 hours) when the tweet has been created. (0-23 UTC hour)
 
   - **tweet_feature_creation_timestamp_week_day**: int:
 <br>The week day when the tweet has been created (0-6 UTC date)
 
 
   - **tweet_feature_creation_timestamp_day_phase**: int:
 <br>The phase of the day when the tweet has been created. It can be NIGHT, MORNING, LUNCH, AFTERNOON or EVENING.
   - **tweet_feature_creation_timestamp_day_phase**: int:
 <br>The shifted phase of the day (+12 hours) when the tweet has been created. It can be NIGHT, MORNING, LUNCH, AFTERNOON or EVENING.


 #### Is engagement type
 **Only for train and local validation test**
 
   - **tweet_feature_engagement_is_like**: bool:
 <br>True if the tweet has been liked by the engager.
   - **tweet_feature_engagement_is_retweet**: bool:
 <br>True if the tweet has been retweeted by the engager.
   - **tweet_feature_engagement_is_comment**: bool:
 <br>True if the tweet has been commented by the engager.
   - **tweet_feature_engagement_is_reply**: bool:
 <br>True if the tweet has been replied by the engager.
   - **tweet_feature_engagement_is_positive**: bool:
 <br>True if the tweet has been involved in a positive engagement by the engager.
   - **tweet_feature_engagement_is_negative**: bool:
 <br>True if the tweet has been involved in a pseudo negative engagement by the engager.
 
 #### Engager knows hashtag
 
   - **engager_feature_knows_hashtag_like**: int:
 <br>The number of time the engager has engaged with a like engagement the hashtags in the tweet.
   - **engager_feature_knows_hashtag_retweet**: int:
 <br>The number of time the engager has engaged with a retweet engagement the hashtags in the tweet.
   - **engager_feature_knows_hashtag_reply**: int:
 <br>The number of time the engager has engaged with a reply engagement the hashtags in the tweet.
   - **engager_feature_knows_hashtag_comment**: int:
 <br>The number of time the engager has engaged with a comment engagement the hashtags in the tweet.
   - **engager_feature_knows_hashtag_negative**: int:
 <br>The number of time the engager has engaged with a negative engagement the hashtags in the tweet.
   - **engager_feature_knows_hashtag_positive**: int:
 <br>The number of time the engager has engaged with a positive engagement the hashtags in the tweet.
 
 
 #### Number of (previous) engagement
 
   - **engager_feature_number_of_previous_like_engagement**: int:
 <br>The number of time the engager has previously engaged a tweet with a like engagement.
   - **engager_feature_number_of_previous_retweet_engagement**: int:
 <br>The number of time the engager has previously engaged a tweet with a retweet engagement.
   - **engager_feature_number_of_previous_reply_engagement**: int:
 <br>The number of time the engager has previously engaged a tweet with a reply engagement.
   - **engager_feature_number_of_previous_comment_engagement**: int:
 <br>The number of time the engager has previously engaged a tweet with a comment engagement.
   - **engager_feature_number_of_previous_positive_engagement**: int:
 <br>The number of time the engager has previously engaged a tweet with a positive engagement.
   - **engager_feature_number_of_previous_negative_engagement**: int:
 <br>The number of time the engager has previously engaged a tweet with a negative engagement.
 
   - **number_of_engagements_like**: int:
 <br>The number of time the engager has engaged a tweet with a like engagement.
   - **number_of_engagements_retweet**: int:
 <br>The number of time the engager has engaged a tweet with a retweet engagement.
   - **number_of_engagements_reply**: int:
 <br>The number of time the engager has engaged a tweet with a reply engagement.
   - **number_of_engagements_comment**: int:
 <br>The number of time the engager has engaged a tweet with a comment engagement.
   - **number_of_engagements_positive**: int:
 <br>The number of time the engager has engaged a tweet with a positive engagement.
   - **number_of_engagements_negative**: int:
 <br>The number of time the engager has engaged a tweet with a negative engagement.
 
 #### Number of engagements ratio
 
   - **number_of_engagements_ratio_like**: int:
 <br>The ratio 'number of previous like engagements'/'number of all previous engagements'. 0 when the user has never been seen.
   - **number_of_engagements_ratio_retweet**: int:
 <br>The ratio 'number of previous retweet engagements'/'number of all previous engagements'. 0 when the user has never been seen.
   - **number_of_engagements_ratio_reply**: int:
 <br>The ratio 'number of previous reply engagements'/'number of all previous engagements'. 0 when the user has never been seen.
   - **number_of_engagements_ratio_comment**: int:
 <br>The ratio 'number of previous comment engagements'/'number of all previous engagements'. 0 when the user has never been seen.
   - **number_of_engagements_ratio_positive**: int:
 <br>The ratio 'number of previous positive engagements'/'number of all previous engagements'. 0 when the user has never been seen.
   - **number_of_engagements_ratio_negative**: int:
 <br>The ratio 'number of previous negative engagements'/'number of all previous engagements'. 0 when the user has never been seen.
 
   - **number_of_engagements_ratio_like_1**: int:
 <br>The ratio 'number of previous like engagements'/'number of all previous engagements'. -1 when the user has never been seen.
   - **number_of_engagements_ratio_retweet_1**: int:
 <br>The ratio 'number of previous retweet engagements'/'number of all previous engagements'. -1 when the user has never been seen.
   - **number_of_engagements_ratio_reply_1**: int:
 <br>The ratio 'number of previous reply engagements'/'number of all previous engagements'. -1 when the user has never been seen.
   - **number_of_engagements_ratio_comment_1**: int:
 <br>The ratio 'number of previous comment engagements'/'number of all previous engagements'. -1 when the user has never been seen.
   - **number_of_engagements_ratio_positive_1**: int:
 <br>The ratio 'number of previous positive engagements'/'number of all previous engagements'. -1 when the user has never been seen.
   - **number_of_engagements_ratio_negative_1**: int:
 <br>The ratio 'number of previous negative engagements'/'number of all previous engagements'. -1 when the user has never been seen.
 
   - **number_of_engagements_ratio_like**: int:
 <br>The ratio 'number of like engagements'/'number of all engagements'.
   - **number_of_engagements_ratio_retweet**: int:
 <br>The ratio 'number of retweet engagements'/'number of all engagements'.
   - **number_of_engagements_ratio_reply**: int:
 <br>The ratio 'number of reply engagements'/'number of all engagements'.
   - **number_of_engagements_ratio_comment**: int:
 <br>The ratio 'number of comment engagements'/'number of all engagements'.
   - **number_of_engagements_ratio_positive**: int:
 <br>The ratio 'number of positive engagements'/'number of all engagements'.
   - **number_of_engagements_ratio_negative**: int:
 <br>The ratio 'number of negative engagements'/'number of all engagements'.
 
 #### Main Language

  - **engager_main_language**: int:
 <br>The main language of the engager.
  - **creator_main_language**: int:
 <br>The main language of the creator.
  - **creator_and_engager_have_same_main_language**: int:
 <br>True if the creator and the engager have the same main language.
   - **is_tweet_in_creator_main_language**: int:
 <br>True if the tweet is in the creator main language.
   - **is_tweet_in_engager_main_language**: int:
 <br>True if the tweet is in the engager main language.
   - **is_tweet_in_engager_main_language**: int:
 <br>True if the tweet is in the engager main language.
   - **statistical_probability_main_language_of_engager_engage_tweet_language_1**: int:
 <br>Statical data explaining how probable a user that have a certain language know also the tweet language. (Excluding the relation language_X - language_X)
   - **statistical_probability_main_language_of_engager_engage_tweet_language_2**: int:
 <br>Statical data explaining how probable a user that have a certain language know also the tweet language. (Including the relation language_X - language_X)
