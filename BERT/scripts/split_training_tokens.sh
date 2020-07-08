
mkdir -p tweet_tokens/subsample/tweet_tokens/day_1

cd tweet_tokens/subsample/tweet_tokens/day_1

split -d --lines=1000000 --suffix-length=2 --additional-suffix=".csv" ../../text_tokens_clean_days_1_unique.csv text_tokens_clean_days_1_unique_

