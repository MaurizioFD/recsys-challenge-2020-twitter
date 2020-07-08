import pandas as pd
import RootPath
import gzip
import pathlib as pl
import random
import codecs
from Utils.Data.Data import get_dataset_xgb, get_feature
STARTING_TIME = 1580947200
SECONDS_IN_A_DAY = 86400
SECONDS_IN_A_WEEK = SECONDS_IN_A_DAY * 7

def holdout_split_train_test(input_dataset_id: str, pc_hold_out: float = 0.30, random_seed = 888):
    input_file_name = f"{RootPath.get_dataset_path()}/{input_dataset_id}.csv.gz"
    train_file_name = f"{RootPath.get_dataset_path()}/cherry_train.csv.gz"
    val_file_name = f"{RootPath.get_dataset_path()}/cherry_val.csv.gz"


    train_file = gzip.open(train_file_name, "wb")
    test_file = gzip.open(val_file_name, "wb")

    x = get_feature("mapped_feature_engager_id", "holdout_new_train")
    y = x.groupby("mapped_feature_engager_id").size()
    a_0 = y[y == 0]
    a_1 = y[y == 1]
    a_2 = y[y == 2]
    a_3 = y[y == 3]

    line_counter = 0

    list_1 = x[x['mapped_feature_engager_id'].isin(set(a_1.sample(469531).index))].index.tolist()
    list_1 += x[x['mapped_feature_engager_id'].isin(set(a_2.sample(550118).index))].index.tolist()
    list_1 += x[x['mapped_feature_engager_id'].isin(set(a_3.sample(313919).index))].index.tolist()

    lines_to_val = set(list_1)

    r = random.Random(random_seed)

    r.random()
    with gzip.open(input_file_name, "rb") as file:
        for line in file:
            if r.random() >= pc_hold_out:
                if line_counter in lines_to_val:
                    a, b, c, d, e, f, g, h, timestamp, j = line.decode('utf-8').split("\x01", 9)
                    timestamp = int(timestamp)
                    timestamp += SECONDS_IN_A_WEEK
                    line = codecs.encode(a+"\x01"+b+"\x01"+c+"\x01"+d+"\x01"+e+"\x01"+f+"\x01"+g+"\x01"+h+"\x01"+str(timestamp)+"\x01"+j,encoding='utf-8')
                    test_file.write(line)
                else:
                    train_file.write(line)
                line_counter += 1
            else:
                if line_counter in lines_to_val:
                    print("Oh oh oopsie woopsie. this is not an error")
                a, b, c, d, e, f, g, h, timestamp, j = line.decode('utf-8').split("\x01", 9)
                timestamp = int(timestamp)
                timestamp += SECONDS_IN_A_WEEK
                line = codecs.encode(a+"\x01"+b+"\x01"+c+"\x01"+d+"\x01"+e+"\x01"+f+"\x01"+g+"\x01"+h+"\x01"+str(timestamp)+"\x01"+j,encoding='utf-8')
                test_file.write(line)

if __name__ == '__main__':
    holdout_split_train_test("new_train")