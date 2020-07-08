cd $1
mkdir -p training/tsv
#mkdir -p training/parquet
cd training/tsv
split -d --lines=1000000 --suffix-length=2 --filter='gzip > $FILE.gz' ../../training.tsv data_

#cd ../..

#mkdir -p evaluation/tsv
#mkdir -p evaluation/parquet
#cd evaluation/tsv
#split -d --lines=1000000 --suffix-length=2 --filter='gzip > $FILE.gz' ../../val.tsv data_

