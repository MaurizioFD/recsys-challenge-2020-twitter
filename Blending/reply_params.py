
def lgbm_get_params():
    params = []

    params.append({
        'num_iterations': 619,
        'num_leaves': 789,
        'max_depth': 6,
        'lambda_l1': 50.0,
        'lambda_l2': 45.215133554212514,
        'colsample_bynode': 0.43251797168693623,
        'colsample_bytree': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 2,
        'min_data_in_leaf': 545,
    })

    params.append({
        'num_iterations': 396,
        'num_leaves': 1560,
        'max_depth': 6,
        'lambda_l1': 8.906119218844239,
        'lambda_l2': 47.28155841673409,
        'colsample_bynode': 0.49654265986746393,
        'colsample_bytree': 0.6117736410415332,
        'bagging_fraction': 1.0,
        'bagging_freq': 1,
        'min_data_in_leaf': 1713,
    })

    params.append({
        'num_iterations': 546,
        'num_leaves': 20,
        'max_depth': 7,
        'lambda_l1': 21.95252646091185,
        'lambda_l2': 48.509695155242355,
        'colsample_bynode': 0.6955101184114274,
        'colsample_bytree': 0.6213556648043124,
        'bagging_fraction': 1.0,
        'bagging_freq': 1,
        'min_data_in_leaf': 468,
    })

    params.append({
        'num_iterations': 431,
        'num_leaves': 1852,
        'max_depth': 6,
        'lambda_l1': 35.697394512543376,
        'lambda_l2': 13.254038397905015,
        'colsample_bynode': 0.4,
        'colsample_bytree': 0.8023619678582585,
        'bagging_fraction': 1.0,
        'bagging_freq': 10,
        'min_data_in_leaf': 468,
    })

    params.append({
        'num_iterations': 171,
        'num_leaves': 827,
        'max_depth': 11,
        'lambda_l1': 20.835015524724298,
        'lambda_l2': 49.14299360409988,
        'colsample_bynode': 0.9469113483943965,
        'colsample_bytree': 0.5123700947463752,
        'bagging_fraction': 0.7352547050374744,
        'bagging_freq': 6,
        'min_data_in_leaf': 1458,
    })

    params.append({
        'num_iterations': 230,
        'num_leaves': 103,
        'max_depth': 32,
        'lambda_l1': 21.886706736695974,
        'lambda_l2': 31.375418366415943,
        'colsample_bynode': 0.8540113841121874,
        'colsample_bytree': 0.41926801219632537,
        'bagging_fraction': 0.8633652202367414,
        'bagging_freq': 6,
        'min_data_in_leaf': 1586,
    })

    return params


def xgb_get_params():
    params = []

    return params

