import pandas as pd
import RootPath
import gzip
import pathlib as pl

STARTING_TIME = 1580947200
SECONDS_IN_A_DAY = 86400


def _check_and_write(
        train_file,
        test_file,
        train_set_from_time,
        train_set_to_time,
        test_set_from_time,
        test_set_to_time,
        line,
        timestamp
):
    if train_set_from_time <= timestamp < train_set_to_time:
        train_file.write(line)
    elif test_set_from_time <= timestamp < test_set_to_time:
        test_file.write(line)


def split_train_val(
        train_filename: str,
        test_filename: str,
        train_days: int = 1,
        test_days: int = 1):
    train_set_from_time = STARTING_TIME
    train_set_to_time = STARTING_TIME + train_days * SECONDS_IN_A_DAY
    test_set_from_time = STARTING_TIME + train_days * SECONDS_IN_A_DAY
    test_set_to_time = STARTING_TIME + (train_days + test_days) * SECONDS_IN_A_DAY

    train_filename = RootPath.get_dataset_path().joinpath(f"{train_filename}.csv.gz")
    test_filename = RootPath.get_dataset_path().joinpath(f"{test_filename}.csv.gz")
    input_filename = RootPath.get_dataset_path().joinpath("train.csv.gz")

    assert not pl.Path(train_filename).exists(), "file already exists"
    assert not pl.Path(test_filename).exists(), "file already exists"

    train_file = gzip.open(train_filename, 'wb')
    test_file = gzip.open(test_filename, 'wb')

    with gzip.open(input_filename, "rb") as file:
        for line in file:
            _, _, _, _, _, _, _, _, timestamp, _ = line.decode('utf-8').split("\x01", 9)
            timestamp = int(timestamp)
            _check_and_write(
                train_file,
                test_file,
                train_set_from_time,
                train_set_to_time,
                test_set_from_time,
                test_set_to_time,
                line,
                timestamp
            )


def split_train_val_multiple(
        train_filename_list: list,
        test_filename_list: list,
        train_days_list: list,
        test_days_list: list):
    assert len(train_filename_list) == len(test_filename_list)
    assert len(train_filename_list) == len(train_days_list)
    assert len(train_filename_list) == len(test_days_list)

    train_set_from_time_list = [STARTING_TIME for _ in train_days_list]
    train_set_to_time_list = [STARTING_TIME + train_days * SECONDS_IN_A_DAY for train_days in train_days_list]
    test_set_from_time_list = [STARTING_TIME + train_days * SECONDS_IN_A_DAY for train_days in train_days_list]
    test_set_to_time_list = [STARTING_TIME + (train_days+test_days) * SECONDS_IN_A_DAY for train_days, test_days in
                             zip(train_days_list, test_days_list)]

    input_filename = RootPath.get_dataset_path().joinpath("train.csv.gz")

    train_filename_list = [RootPath.get_dataset_path().joinpath(f"{train_filename}.csv.gz") for train_filename in train_filename_list]
    test_filename_list = [RootPath.get_dataset_path().joinpath(f"{test_filename}.csv.gz") for test_filename in test_filename_list]

    assert all([not pl.Path(train_filename).exists() for train_filename in train_filename_list]), "files already exist"
    assert all([not pl.Path(test_filename).exists() for test_filename in test_filename_list]), "files already exist"

    train_file_list = [gzip.open(train_filename, 'wb') for train_filename in train_filename_list]
    test_file_list = [gzip.open(test_filename, 'wb') for test_filename in test_filename_list]

    zipped_list = zip(
        train_file_list,
        test_file_list,
        train_set_from_time_list,
        train_set_to_time_list,
        test_set_from_time_list,
        test_set_to_time_list
    )

    with gzip.open(input_filename, "rb") as file:
        for line in file:
            _, _, _, _, _, _, _, _, timestamp, _ = line.decode('utf-8').split("\x01", 9)
            timestamp = int(timestamp)
            for i in range(len(train_filename_list)):
                _check_and_write(
                    train_file_list[i],
                    test_file_list[i],
                    train_set_from_time_list[i],
                    train_set_to_time_list[i],
                    test_set_from_time_list[i],
                    test_set_to_time_list[i],
                    line,
                    timestamp
                )
