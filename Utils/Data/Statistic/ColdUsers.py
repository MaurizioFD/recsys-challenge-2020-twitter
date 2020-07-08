from Utils.Data.DatasetUtils import get_test_or_val_set_id_from_train
from Utils.Data import Data
from Utils.Data.Data import get_dataset_xgb, get_feature
from Utils.Data.DataUtils import create_all
from Utils.Data.Features.Generated.EngagerFeature.KnownEngagementCount import *
from Utils.Data.Features.MappedFeatures import MappedFeatureCreatorId
from Utils.Data.Split import TimestampBasedSplit
import pandas as pd
import numpy as np


def analyze_cold_engager(dataset_id: str):
    train_dataset = dataset_id
    test_dataset = get_test_or_val_set_id_from_train(dataset_id)

    is_positive = TweetFeatureEngagementIsPositive(train_dataset)
    is_positive_df = is_positive.load_or_create()
    positive_mask = is_positive_df[is_positive.feature_name]

    train_creator_id = MappedFeatureCreatorId(train_dataset)
    train_engager_id = MappedFeatureEngagerId(train_dataset)

    train_creator_id_df = train_creator_id.load_or_create()[positive_mask]
    train_engager_id_df = train_engager_id.load_or_create()[positive_mask]

    train_users_id_df = pd.DataFrame(pd.concat(
        [
            train_creator_id_df[train_creator_id.feature_name],
            train_engager_id_df[train_engager_id.feature_name]
        ],
        axis=0
    ).unique())

    train_users_id_df['is_train'] = True
    train_users_id_df.set_index(train_users_id_df.columns[0], inplace=True)

    # test_creator_id = MappedFeatureCreatorId("test")
    test_engager_id = MappedFeatureEngagerId(test_dataset)

    # test_creator_id_df = test_creator_id.load_or_create()
    test_engager_id_df = test_engager_id.load_or_create()

    total_number_of_engagements = len(test_engager_id_df)

    test_users_id_df = pd.DataFrame(pd.concat(
        [
            # test_creator_id_df[test_creator_id.feature_name],
            test_engager_id_df[test_engager_id.feature_name]
        ],
        axis=0
    ).unique())

    count = pd.DataFrame({'count': test_users_id_df.groupby(test_users_id_df.columns[0]).size()})

    test_users_id_df['is_test'] = True
    test_users_id_df.set_index(test_users_id_df.columns[0], inplace=True)

    x = train_users_id_df.join(test_users_id_df, how='outer')
    x = x.fillna(False)

    train_mask = x['is_train'] == False
    test_mask = x['is_test'] == True

    mask = train_mask & test_mask

    cold_users = np.array(x[mask].index.array)
    print(f"------------------------")
    print(dataset_id)
    print(f"Unique test engagers are: {len(test_users_id_df)}")
    print(f"Unique test cold engagers are: {len(cold_users)}")
    print(f"Engagements in test set are: {total_number_of_engagements}")
    print(f"Engagements of cold engagers in test set are: {count['count'][mask].sum()}")
    print(f"Man number of engagement per cold engager in test set are: {count['count'][mask].max()}")
    print(f"Probability that an engagement is engaged by a cold engagers: {len(cold_users)/total_number_of_engagements}")
    print(f"------------------------")



def analyze_cold_creator(dataset_id: str):
    train_dataset = dataset_id
    test_dataset = get_test_or_val_set_id_from_train(dataset_id)

    is_positive = TweetFeatureEngagementIsPositive(train_dataset)
    is_positive_df = is_positive.load_or_create()
    positive_mask = is_positive_df[is_positive.feature_name]

    train_creator_id = MappedFeatureCreatorId(train_dataset)
    train_engager_id = MappedFeatureEngagerId(train_dataset)

    train_creator_id_df = train_creator_id.load_or_create()[positive_mask]
    train_engager_id_df = train_engager_id.load_or_create()[positive_mask]

    train_users_id_df = pd.DataFrame(pd.concat(
        [
            train_creator_id_df[train_creator_id.feature_name],
            train_engager_id_df[train_engager_id.feature_name]
        ],
        axis=0
    ).unique())

    train_users_id_df['is_train'] = True
    train_users_id_df.set_index(train_users_id_df.columns[0], inplace=True)

    test_creator_id = MappedFeatureCreatorId(test_dataset)
    # test_engager_id = MappedFeatureEngagerId(test_dataset)

    test_creator_id_df = test_creator_id.load_or_create()
    # test_engager_id_df = test_engager_id.load_or_create()

    total_number_of_engagements = len(test_creator_id_df)

    test_users_id_df = pd.DataFrame(pd.concat(
        [
            test_creator_id_df[test_creator_id.feature_name],
            # test_engager_id_df[test_engager_id.feature_name]
        ],
        axis=0
    ).unique())

    count = pd.DataFrame({'count': test_users_id_df.groupby(test_users_id_df.columns[0]).size()})

    test_users_id_df['is_test'] = True
    test_users_id_df.set_index(test_users_id_df.columns[0], inplace=True)

    x = train_users_id_df.join(test_users_id_df, how='outer')
    x = x.fillna(False)

    train_mask = x['is_train'] == False
    test_mask = x['is_test'] == True

    mask = train_mask & test_mask

    cold_users = np.array(x[mask].index.array)
    print(f"------------------------")
    print(dataset_id)
    print(f"Unique test creators are: {len(test_users_id_df)}")
    print(f"Unique test cold creators are: {len(cold_users)}")
    print(f"Engagements in test set are: {total_number_of_engagements}")
    print(f"Engagements of cold users in test set are: {count['count'][mask].sum()}")
    print(f"Man number of engagement per cold creator in test set are: {count['count'][mask].max()}")
    print(f"Probability that an engagement is engaged by a cold creators: {len(cold_users)/total_number_of_engagements}")
    print(f"------------------------")