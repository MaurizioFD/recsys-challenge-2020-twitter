import pandas as pd
import RootPath
import gzip
import pathlib as pl
import random
import codecs

STARTING_TIME = 1580947200
SECONDS_IN_A_DAY = 86400
SECONDS_IN_A_WEEK = SECONDS_IN_A_DAY * 7

def holdout_split_train_test(input_dataset_id: str, pc_hold_out: float = 0.30, random_seed = 888):
    input_file_name = f"{RootPath.get_dataset_path()}/{input_dataset_id}.csv.gz"
    train_file_name = f"{RootPath.get_dataset_path()}/holdout_new_train.csv.gz"
    val_file_name = f"{RootPath.get_dataset_path()}/holdout_new_test.csv.gz"


    train_file = gzip.open(train_file_name, "wb")
    test_file = gzip.open(val_file_name, "wb")

    r = random.Random(random_seed)

    r.random()
    with gzip.open(input_file_name, "rb") as file:
        for line in file:
            if r.random() >= pc_hold_out:
                train_file.write(line)
            else:
                a, b, c, d, e, f, g, h, timestamp, j = line.decode('utf-8').split("\x01", 9)
                timestamp = int(timestamp)
                timestamp += SECONDS_IN_A_WEEK
                line = codecs.encode(a+"\x01"+b+"\x01"+c+"\x01"+d+"\x01"+e+"\x01"+f+"\x01"+g+"\x01"+h+"\x01"+str(timestamp)+"\x01"+j,encoding='utf-8')
                test_file.write(line)


if __name__ == '__main__':
    holdout_split_train_test("new_train")