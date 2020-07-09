mkdir -p Dataset
gzip -c training.tsv > ./Dataset/new_train.csv.gz
gzip -c val.tsv > ./Dataset/new_test.csv.gz
gzip -c test.tsv > ./Dataset/last_test.csv.gz
