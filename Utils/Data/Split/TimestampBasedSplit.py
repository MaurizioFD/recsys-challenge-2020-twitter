import random
import RootPath
import gzip

def split_with_timestamp(input_dataset_id: str, pc_hold_out: float = 0.75, timestamp_threshold = 1581465600, random_seed = 888):
    input_file_name = f"{RootPath.get_dataset_path()}/{input_dataset_id}.csv.gz"
    train_file_name = f"{RootPath.get_dataset_path()}/train_split_with_timestamp_from_{input_dataset_id}_random_seed_{random_seed}_timestamp_threshold_{timestamp_threshold}_holdout_{int(pc_hold_out * 100)}.csv.gz"
    val_file_name = f"{RootPath.get_dataset_path()}/val_split_with_timestamp_from_{input_dataset_id}_random_seed_{random_seed}_timestamp_threshold_{timestamp_threshold}_holdout_{int(pc_hold_out * 100)}.csv.gz"

    train_file = gzip.open(train_file_name, "wb")
    test_file = gzip.open(val_file_name, "wb")

    r = random.Random(random_seed)

    r.random()

    with gzip.open(input_file_name, "rb") as file:
        for line in file:
            if r.random() < pc_hold_out:
                _, _, _, _, _, _, _, _, timestamp, _ = line.decode('utf-8').split("\x01", 9)
                if int(timestamp) <= timestamp_threshold:
                    train_file.write(line)
                else:
                    test_file.write(line)
