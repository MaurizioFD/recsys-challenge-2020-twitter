gzip -c training.tsv > ./Dataset/train.csv.gz
gzip -c val.tsv > ./Dataset/new_test.csv.gz
gzip -c test.tsv > ./Dataset/last_test.csv.gz