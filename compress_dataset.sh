mkdir -p Dataset
gzip -c training.tsv > ./Dataset/new_train.csv.gz
gzip -c val.tsv > ./Dataset/new_test.csv.gz
gzip -c test.tsv > ./Dataset/last_test.csv.gz
cp ./Dataset/new_train.csv.gz ./Dataset/train.csv.gz
cp ./Dataset/new_test.csv.gz ./Dataset/test.csv.gz