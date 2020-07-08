
def lgbm_get_params():
    params = []
    '''
    
    LGBM-like-freschi-sovracampione
    ITERATION NUMBER 18
    
    num_leaves= 2782
    
    learning rate= 0.047269495547881867
    
    max_depth= 69
    
    lambda_l1= 11.33858475880979
    
    lambda_l2= 1.0
    
    colsample_bynode= 0.7703588615883736
    
    colsample_bytree= 0.9389031548846904
    
    bagging_fraction= 1.0
    
    bagging_freq= 8
    
    max_bin= 5000
    
    min_data_in_leaf= 1171
    LGBM-like-freschi-sovracampione
    -------
    EXECUTION TIME: 7319.081578731537
    -------
    best_es_iteration: 1000
    -------
    PRAUC = 0.8240523857778829
    RCE   = 33.35315963830989
    '''

    params.append({
        'objective': "binary",

        'num_threads': -1,

        'num_iterations': 1000,

        'num_leaves': 2782,

        'learning_rate': 0.047269495547881867,

        'max_depth': 69,

        'lambda_l1': 11.33858475880979,

        'lambda_l2': 1.0,

        'colsample_bynode': 0.7703588615883736,

        'colsample_bytree': 0.9389031548846904,

        'bagging_fraction': 1.0,

        'bagging_freq': 8,

        'max_bin': 5000,

        'min_data_in_leaf': 1171,

        # 'early_stopping_rounds': 15
    })

    '''
    LGBM-like-freschi-sovracampione
    ITERATION NUMBER 19
    
    num_leaves= 2644
    
    learning rate= 0.2556629498107492
    
    max_depth= 69
    
    lambda_l1= 20.144135467449093
    
    lambda_l2= 1.1162458802315145
    
    colsample_bynode= 0.6572831988379086
    
    colsample_bytree= 0.9032639334840711
    
    bagging_fraction= 0.9266574927996606
    
    bagging_freq= 5
    
    max_bin= 5000
    
    min_data_in_leaf= 400
    
    LGBM-like-freschi-sovracampione
    -------
    EXECUTION TIME: 2594.5206253528595
    -------
    best_es_iteration: 358
    -------
    PRAUC = 0.8200380095572581
    RCE   = 32.82785871179856
    '''
    params.append({
        'objective': "binary",

        'num_threads': -1,

        'num_iterations': 358,

        'num_leaves': 2644,

        'learning_rate': 0.2556629498107492,

        'max_depth': 69,

        'lambda_l1': 20.144135467449093,

        'lambda_l2': 1.1162458802315145,

        'colsample_bynode': 0.6572831988379086,

        'colsample_bytree': 0.9032639334840711,

        'bagging_fraction': 0.9266574927996606,

        'bagging_freq': 5,

        'max_bin': 5000,

        'min_data_in_leaf': 400,

        # 'early_stopping_rounds': 15
    })


    '''
    LGBM-like-freschi-sovracampione
    ITERATION NUMBER 21
    
    num_leaves= 1793
    
    learning rate= 0.02927778518031793
    
    max_depth= 38
    
    lambda_l1= 1.0
    
    lambda_l2= 16.606353530667754
    
    colsample_bynode= 0.6153581730566215
    
    colsample_bytree= 1.0
    
    bagging_fraction= 1.0
    
    bagging_freq= 10
    
    max_bin= 5000
    
    min_data_in_leaf= 518
    
    LGBM-like-freschi-sovracampione
    -------
    EXECUTION TIME: 7176.927843809128
    -------
    best_es_iteration: 1000
    -------
    PRAUC = 0.8159999117566297
    RCE   = 31.914442137394527
    '''

    params.append({
        'objective': "binary",

        'num_threads': -1,

        'num_iterations': 1000,

        'num_leaves': 1793,

        'learning_rate': 0.02927778518031793,

        'max_depth': 38,

        'lambda_l1': 1.0,

        'lambda_l2': 16.606353530667754,

        'colsample_bynode': 0.6153581730566215,

        'colsample_bytree': 1.0,

        'bagging_fraction': 1.0,

        'bagging_freq': 10,

        'max_bin': 5000,

        'min_data_in_leaf': 518,

        # 'early_stopping_rounds': 15
    })

    '''
    LGBM-like-freschi-sovracampione
    ITERATION NUMBER 10
    
    num_leaves= 839
    
    learning rate= 0.11197799544768677
    
    max_depth= 64
    
    lambda_l1= 17.09966070314015
    
    lambda_l2= 8.766381416248917
    
    colsample_bynode= 0.8477557541386376
    
    colsample_bytree= 0.8461274389450735
    
    bagging_fraction= 0.553969305344419
    
    bagging_freq= 8
    
    max_bin= 2573
    
    min_data_in_leaf= 779
    
    LGBM-like-freschi-sovracampione
    -------
    EXECUTION TIME: 3496.339388847351
    -------
    best_es_iteration: 962
    -------
    PRAUC = 0.8142445328001886
    RCE   = 31.66096325701592
    '''

    params.append({
        'objective': "binary",

        'num_threads': -1,

        'num_iterations': 962,

        'num_leaves': 839,

        'learning_rate': 0.11197799544768677,

        'max_depth': 64,

        'lambda_l1': 17.09966070314015,

        'lambda_l2': 8.766381416248917,

        'colsample_bynode': 0.8477557541386376,

        'colsample_bytree': 0.8461274389450735,

        'bagging_fraction': 0.553969305344419,

        'bagging_freq': 8,

        'max_bin': 2573,

        'min_data_in_leaf': 779,

        # 'early_stopping_rounds': 15
    })



    return params


def xgb_get_params():
    params = []

    return params

