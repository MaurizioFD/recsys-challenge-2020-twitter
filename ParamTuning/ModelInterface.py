import functools
import sys
import os.path
from Models.GBM.XGBoost import XGBoost
from Models.GBM.LightGBM import LightGBM
from Models.GBM.CatBoost import CatBoost
from Utils.Eval.Metrics import ComputeMetrics as CoMe
from Utils.Base.ParamRangeDict import xgbRange
from Utils.Base.ParamRangeDict import xgbName
from Utils.Base.ParamRangeDict import lgbmRange
from Utils.Base.ParamRangeDict import lgbmName
from Utils.Base.ParamRangeDict import lgbmRangeCold
from Utils.Base.ParamRangeDict import lgbmNameCold
from Utils.Base.ParamRangeDict import catRange
from Utils.Base.ParamRangeDict import catName
from Utils.Data.Data import get_dataset_xgb
from Utils.Data.Data import get_dataset_xgb_batch
from Utils.Data.DataUtils import TRAIN_IDS,  TEST_IDS
import pandas as pd
import datetime as dt
import time
from tqdm import tqdm
import xgboost as xgb
import numpy as np
import multiprocessing as mp
import xgboost as xgb
import catboost as cat
import lightgbm as lgbm
from Utils.TelegramBot import telegram_bot_send_update

class ModelInterface(object):
    def __init__(self, model_name, kind, mode):
        self.model_name = model_name
        self.kind = kind
        self.mode = mode
        # Datasets
        self.test = None
        self.train = None
        self.val = None
        #Batch datasets
        self.val_id = None
        #NCV early stopping param
        self.es_ncv = False
        # LOGS PARAMS
        # Counter of iterations
        self.iter_count = 1
        # Filename for logs
        self.path = None
        # True make logs, false don't
        self.make_log = True
        # Parameters to set
        #------------XGB----------------
        self.verbosity=1
        self.process_type="default"
        self.tree_method="auto"
        self.objective="binary:logistic"
        self.num_parallel_tree=4
        self.eval_metric="rmsle"
        self.early_stopping_rounds=5
        #-------------------------------
        #------------CAT----------------
        # Not in tuning dict
        self.boosting_type = "Plain"
        self.model_shrink_mode = "Constant"
        self.leaf_estimation_method = "Newton"
        self.bootstrap_type = "Bernoulli"
        self.categorical_features = None
        # Initialized above no need to do it again
        # self.early_stopping_rounds=5
        # self.verbosity
        #-------------------------------
        
        
#------------------------------------------------------
#                   SINGLE TRAIN
#------------------------------------------------------
    #Score function for the XGBoost model
    def blackBoxXGB(self, param):
        #Log time
        start_log_time = time.time()
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = XGBoost(kind=self.kind,
                        #Not in tuning dict
                        verbosity=self.verbosity,
                        process_type=self.process_type,
                        tree_method=self.tree_method,
                        objective=self.objective,
                        eval_metric=self.eval_metric,
                        early_stopping_rounds=self.early_stopping_rounds,
                        #In tuning dict
                        num_rounds=       param[0],
                        max_depth=        param[1],
                        min_child_weight= param[2],
                        colsample_bytree= param[3],
                        learning_rate=    param[4],
                        reg_alpha=        param[5],
                        reg_lambda=       param[6],
                        scale_pos_weight= param[7],
                        gamma=            param[8],                        
                        subsample=        param[9],
                        base_score=       param[10],
                        max_delta_step=   param[11],
                        num_parallel_tree=param[12])
        #Training on custom set
        if (self.train is None):
            print("No train set passed to the model.")
        else:
            #dmat_train = self.getDMat(self.X_train, self.Y_train) #------------------------------------- DMATRIX GENERATOR
            model.fit(self.train, self.val)            
            if self.val is not None:
                best_iter = model.getBestIter()
            else:
                best_iter = -1

        #Evaluating on custom set
        if (self.test is None):
            print("No test set provided.")
        else:
            #dmat_test = self.getDMat(self.X_test, self.Y_test) #------------------------------------- DMATRIX GENERATOR
            prauc, rce, confmat, max_pred, min_pred, avg = model.evaluate(self.test)

        del model
        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(best_iter,
                         prauc, 
                         rce, 
                         confmat, 
                         max_pred, 
                         min_pred, 
                         avg,
                         start_log_time)
        
        #Returning the dumbly combined scores
        if max_pred != min_pred:
            return self.metriComb(prauc, rce)
        else:
            return 1000


#------------------------------------------------------
#                   BATCH TRAIN
#------------------------------------------------------
#       Use the self.batchLoadSets method
#         In order to load the datasets
#------------------------------------------------------
# Batch may be of different sizes, but considering how
# they're distributed giving them the same weight in
# averaging process shouldn't raise problems.
#------------------------------------------------------
    #Score function for the XGBoost model
    def blackBoxXgbBatch(self, param):
        #Log time
        start_log_time = time.time()
        #Saving parameters
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = XGBoost(kind=self.kind,
                        #Not in tuning dict
                        verbosity=self.verbosity,
                        process_type=self.process_type,
                        tree_method=self.tree_method,
                        objective=self.objective,
                        eval_metric=self.eval_metric,
                        early_stopping_rounds=self.early_stopping_rounds,
                        #In tuning dict
                        num_rounds=        param[0],
                        max_depth=         param[1],
                        min_child_weight=  param[2],
                        colsample_bytree=  param[3],
                        learning_rate=     param[4],
                        reg_alpha=         param[5],
                        reg_lambda=        param[6],
                        scale_pos_weight=  param[7],
                        gamma=             param[8],                        
                        subsample=         param[9],
                        base_score=        param[10],
                        max_delta_step=    param[11],
                        num_parallel_tree=param[12])

        best_iter = []
        #Batch train
        for split in tqdm(range(self.tot_train_split)):
            X, Y = get_dataset_xgb_batch(self.tot_train_split, 
                                         split, 
                                         self.train_id, 
                                         self.x_label, 
                                         self.y_label)
            #start_time_training_data = time.time()
            dmat_train = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
            del X, Y

            if self.val_id is not None:
                X, Y = get_dataset_xgb_batch(self.tot_train_split, 
                                             split, 
                                             self.val_id, 
                                             self.x_label, 
                                             self.y_label)
                dmat_val = self.getDMat(X, Y)
                del X, Y
            else:
                dmat_val = None

            #Multistage model fitting
            model.fit(dmat_train, dmat_val)
            del dmat_train

            #Get best iteration obtained with es
            if dmat_val is not None:
                best_iter.append(model.getBestIter())
            else:
                best_iter = -1
            del dmat_val


        #Initializing variables
        tot_prauc = 0
        tot_rce = 0
        tot_confmat = [[0,0],[0,0]]
        max_pred = 0 #Max set to the minimum
        min_pred = 1 #Min set to the maximum
        avg = 0
        #Batch evaluation
        for split in tqdm(range(self.tot_test_split)):
            #Iteratively fetching the dataset
            X, Y = get_dataset_xgb_batch(self.tot_test_split, 
                                         split, 
                                         self.test_id, 
                                         self.x_label, 
                                         self.y_label)

            dmat_test = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
            del X, Y
            #Multistage evaluation
            prauc, rce, confmat, max_tmp, min_tmp, avg_tmp= model.evaluate(dmat_test)
            del dmat_test

            #Summing all the evaluations
            tot_prauc = tot_prauc + prauc
            tot_rce = tot_rce + rce
            
            #Computing some statistics for the log
            if self.make_log is True:
                # Getting maximum over iteration
                if max_tmp > max_pred:
                    max_pred = max_tmp
                # Getting minimum over iteration
                if min_tmp < min_pred:
                    min_pred = min_tmp
                # Getting average over itaration
                avg += avg_tmp
                # Computing confusion matrix
                tot_confmat = tot_confmat + confmat
        del model          

        #Averaging the evaluations over # of validation splits
        tot_prauc = tot_prauc/self.tot_test_split
        tot_rce = tot_rce/self.tot_test_split
        avg = avg/self.tot_test_split

        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(best_iter,
                         tot_prauc, 
                         tot_rce, 
                         tot_confmat, 
                         max_pred, 
                         min_pred, 
                         avg,
                         start_log_time)
        
        #Returning the dumbly combined scores
        return self.metriComb(tot_prauc, tot_rce)


#------------------------------------------------------
#        NESTED CROSS VALIDATION TRAIN
#------------------------------------------------------
#        Use the self.ncvLoadSets method
#         In order to load the datasets
#------------------------------------------------------
# Batch may be of different sizes, but considering how
# they're distributed giving them the same weight in
# averaging process shouldn't raise problems.
#------------------------------------------------------
    #Score function for the XGBoost model
    def blackBoxXgbNCV(self, param):
        #Log time
        start_log_time = time.time()
        #print(param)
        #Saving parameters
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = XGBoost(kind=self.kind,
                        #Not in tuning dict
                        verbosity=self.verbosity,
                        process_type=self.process_type,
                        tree_method=self.tree_method,
                        objective=self.objective,
                        eval_metric=self.eval_metric,
                        early_stopping_rounds=self.early_stopping_rounds,
                        #In tuning dict
                        num_rounds =       param[0],
                        max_depth =        param[1],
                        min_child_weight = param[2],
                        colsample_bytree=  param[3],
                        learning_rate=     param[4],
                        reg_alpha=         param[5],
                        reg_lambda=        param[6],
                        scale_pos_weight=  param[7],
                        gamma=             param[8],                        
                        subsample=         param[9],
                        base_score=        param[10],
                        max_delta_step=    param[11],
                        num_parallel_tree=param[12])

        #Iterable returns pair of train - val sets
        id_pairs = zip(TRAIN_IDS, TEST_IDS)

        #Initializing variables
        weight_factor = 0
        averaging_factor = 0
        tot_prauc = 0
        tot_rce = 0
        tot_confmat = [[0,0],[0,0]]
        max_pred = 0
        min_pred = 1
        avg = 0
        best_iter = []
        #Making iterative train-validations
        for dataset_ids in id_pairs:
            weight_factor += weight_factor+1
            averaging_factor += weight_factor
            #Fetching train set
            X, Y = get_dataset_xgb(dataset_ids[0], 
                                   self.x_label, 
                                   self.y_label)
            
            dmat_train = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
            del X, Y
            # If early stopping is true for ncv fetching validation set
            # by splitting in two the test set with batch method
            if self.es_ncv is True:
                #Fetching val set 
                X, Y = get_dataset_xgb_batch(2, 0,dataset_ids[1], 
                                             self.x_label, 
                                             self.y_label)
                dmat_val = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
                del X, Y
            else:
                dmat_val = None
            
            #Multistage model fitting
            model.fit(dmat_train, dmat_val)
            del dmat_train
            if dmat_val is not None:
                best_iter.append(model.getBestIter())
            else:
                best_iter = -1
            del dmat_val

            if self.es_ncv is True:
                #Fetching test set 
                X, Y = get_dataset_xgb_batch(2, 1,dataset_ids[1], 
                                             self.x_label, 
                                             self.y_label)
                dmat_test = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
                del X, Y    
            else:
                X, Y = get_dataset_xgb(dataset_ids[1], 
                                       self.x_label,
                                       self.y_label)
                dmat_test = self.getDMat(X, Y) #------------------------------------- DMATRIX GENERATOR
                del X, Y
            #Multistage evaluation
            prauc, rce, confmat, max_tmp, min_tmp, avg_tmp= model.evaluate(dmat_test)
            del dmat_test

            #Weighting scores (based on how many days are in the train set)
            tot_prauc += prauc * weight_factor
            tot_rce += rce * weight_factor

            #Computing some statistics for the log
            if self.make_log is True:
                #Getting maximum over iteration
                if max_tmp > max_pred:
                    max_pred = max_tmp
                #Getting minimum over iteration
                if min_tmp < min_pred:
                    min_pred = min_tmp
                #Getting average over itaration
                avg += avg_tmp
                #Computing the confusion matrix
                tot_confmat = tot_confmat + confmat
        del model

        #Averaging scores
        tot_prauc /= averaging_factor
        tot_rce /= averaging_factor
        #Averaging average (lol)
        avg /= len(TRAIN_IDS)

        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(best_iter,
                         tot_prauc, 
                         tot_rce, 
                         tot_confmat, 
                         max_pred, 
                         min_pred, 
                         avg,
                         start_log_time)
        
        #Returning the dumbly combined scores
        return self.metriComb(tot_prauc, tot_rce)

# ------------------------------------------------------
#        XGB BATCH WITH EXTERNAL MEMORY
# ------------------------------------------------------
#       Use the self.batchLoadSetsWithExtMemory method
#         In order to load the datasets
# ------------------------------------------------------
# Batch may be of different sizes, but considering how
# they're distributed giving them the same weight in
# averaging process shouldn't raise problems.
# ------------------------------------------------------
# Score function for the XGBoost model
    def blackBoxXgbBatchExtMem(self, param):
        queue = mp.Queue()
        sub_process = mp.Process(target=run_xgb_external_memory, args=(param, self, queue))
        sub_process.start()
        sub_process.join()
        return queue.get()
        # with mp.Pool(1) as pool:
        #     return pool.map(functools.partial(run_xgb_external_memory, model_interface=self), [param])[0]

#-----------------------------------------
#       TODO:Future implementation
#-----------------------------------------
    # Score function for the lightGBM model
    def blackBoxLGB(self, param):
        #Log time
        start_log_time = time.time()
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = LightGBM(kind=self.kind,
                        objective=self.objective,
                        metric=self.eval_metric,
                        #In tuning dict
                        num_leaves=       param[0],
                        learning_rate=    param[1],
                        max_depth=        param[2],
                        lambda_l1=        param[3],
                        lambda_l2=        param[4],
                        colsample_bynode= param[5],
                        colsample_bytree= param[6],
                        bagging_fraction= param[7],
                        #scale_pos_weight= param[11],        #Remember that scale_pos_wiight and is_unbalance are mutually exclusive
                        bagging_freq=     param[8],
                        max_bin =         param[9],
                        min_data_in_leaf= param[10],
                        #Early stopping
                        early_stopping_rounds=self.early_stopping_rounds,
                        is_unbalance=self.is_unbalance
        )
        #Training on custom set
        if (self.Y_train is None):
            print("No train set passed to the model.")
        else:
            if self.X_val is None:
                model.fit(self.X_train, self.Y_train, categorical_feature=self.categorical_features)           #TODO: AGGIUNGI CATEGORICAL FEATURE
                best_iter = -1
            else:
                model.fit(self.X_train, self.Y_train, X_val=self.X_val, Y_val=self.Y_val, categorical_feature=self.categorical_features)
                best_iter = model.get_best_iter()

        #Evaluating on custom set
        if (self.Y_test is None):
            print("No test set provided.")
        else:
            prauc, rce, confmat, max_pred, min_pred, avg = model.evaluate(self.X_test.to_numpy(),self.Y_test.to_numpy())

        del model
        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(best_iter,
                         prauc, 
                         rce, 
                         confmat, 
                         max_pred, 
                         min_pred, 
                         avg,
                         start_log_time)
        
        #Returning the dumbly combined scores
        return self.metriComb(prauc, rce)

#-----------------------------------------
#       ONLY FOR COLD USERS
#-----------------------------------------
    # Score function for the lightGBM model
    def blackBoxLGBCold(self, param):
        #Log time
        start_log_time = time.time()
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = LightGBM(kind=self.kind,
                        objective=self.objective,
                        metric=self.eval_metric,
                        #In tuning dict
                        num_leaves=       param[0],
                        max_depth=        param[1],
                        learning_rate=    0.25,
                        lambda_l1=        param[2],
                        lambda_l2=        param[3],
                        colsample_bynode= param[4],
                        colsample_bytree= param[5],
                        bagging_fraction= param[6],
                        bagging_freq=     param[7],
                        min_data_in_leaf= param[8],
                        #Early stopping
                        early_stopping_rounds=self.early_stopping_rounds,
                        is_unbalance=self.is_unbalance
        )
        #Training on custom set
        if (self.Y_train is None):
            print("No train set passed to the model.")
        else:
            if self.X_val is None:
                model.fit(self.X_train, self.Y_train, categorical_feature=self.categorical_features)           #TODO: AGGIUNGI CATEGORICAL FEATURE
                best_iter = -1
            else:
                model.fit(self.X_train, self.Y_train, X_val=self.X_val, Y_val=self.Y_val, categorical_feature=self.categorical_features)
                best_iter = model.get_best_iter()

        #Evaluating on custom set
        if (self.Y_test is None):
            print("No test set provided.")
        else:
            prauc, rce, confmat, max_pred, min_pred, avg = model.evaluate(self.X_test.to_numpy(),self.Y_test.to_numpy())

        del model
        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(best_iter,
                         prauc, 
                         rce, 
                         confmat, 
                         max_pred, 
                         min_pred, 
                         avg,
                         start_log_time)
        
        #Returning the dumbly combined scores
        return self.metriComb(prauc, rce)

    # Batch ones    
    # Score function for the lightGBM model
    def blackBoxLgbBatch(self, param):
        #TODO: implement this
        return None

    # NCV ones    
    # Score function for the lightGBM model
    def blackBoxLgbNCV(self, param):
        #TODO: implement this
        return None


    #--------------------------------------
    # CatBoost single train optimization
    #-------------------------------------- 
    # Score function for the CatBoost model
    def blackBoxCAT(self, param):
        #Log time
        start_log_time = time.time()
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = CatBoost(kind = self.kind,
                         verbose=self.verbosity,
                         #Not in tuning dict
                         boosting_type = self.boosting_type,
                         model_shrink_mode = self.model_shrink_mode,
                         leaf_estimation_method = self.leaf_estimation_method,
                         bootstrap_type = self.bootstrap_type,
                         #In tuning dict
                         iterations=                 param[0],
                         depth=                      param[1],
                         learning_rate=              param[2],
                         l2_leaf_reg=                param[3],
                         subsample=                  param[4],
                         random_strenght=            param[5],
                         colsample_bylevel=          param[6],
                         leaf_estimation_iterations= param[7],
                         #scale_pos_weight =          param[8],
                         model_shrink_rate =         param[8],
                         # ES parameters
                         early_stopping_rounds = self.early_stopping_rounds)

        #Training on custom set
        if (self.train is None):
            print("No train set passed to the model.")
        else:
            #dmat_train = self.getDMat(self.X_train, self.Y_train) #------------------------------------- DMATRIX GENERATOR
            print("Training the model:")
            model.fit(self.train, self.val)            
            if self.val is not None:
                best_iter = model.getBestIter()
            else:
                best_iter = -1

        #Evaluating on custom set
        if (self.test is None):
            print("No test set provided.")
        else:
            #dmat_test = self.getDMat(self.X_test, self.Y_test) #------------------------------------- DMATRIX GENERATOR
            print("Evaluating the model:")
            prauc, rce, confmat, max_pred, min_pred, avg = model.evaluate(self.test)
        del model

        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(best_iter,
                         prauc, 
                         rce, 
                         confmat, 
                         max_pred, 
                         min_pred, 
                         avg,
                         start_log_time)
        
        #Returning the scores
        return self.metriComb(prauc, rce)


    
    # Score function for the CatBoost model
    def blackBoxCatBatch(self, param):
        #Log time
        start_log_time = time.time()
        #Saving parameters
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = CatBoost(kind = self.kind,
                         verbose=self.verbosity,
                         #Not in tuning dict
                         boosting_type = self.boosting_type,
                         model_shrink_mode = self.model_shrink_mode,
                         leaf_estimation_method = self.leaf_estimation_method,
                         bootstrap_type = self.bootstrap_type,
                         #In tuning dict
                         iterations=                 param[0],
                         depth=                      param[1],
                         learning_rate=              param[2],
                         l2_leaf_reg=                param[3],
                         subsample=                  param[4],
                         random_strenght=            param[5],
                         colsample_bylevel=          param[6],
                         leaf_estimation_iterations= param[7],
                         #scale_pos_weight =          param[8],
                         model_shrink_rate =         param[8],
                         # ES parameters
                         early_stopping_rounds = self.early_stopping_rounds)

        best_iter = []
        #Batch train
        for split in tqdm(range(self.tot_train_split)):
            X, Y = get_dataset_xgb_batch(self.tot_train_split, 
                                         split, 
                                         self.train_id, 
                                         self.x_label, 
                                         self.y_label)
            #start_time_training_data = time.time()
            pool_train = self.getPool(X, Y) #------------------------------------- DMATRIX GENERATOR
            del X, Y

            if self.val_id is not None:
                X, Y = get_dataset_xgb_batch(self.tot_train_split, 
                                             split, 
                                             self.val_id, 
                                             self.x_label, 
                                             self.y_label)
                pool_val = self.getPool(X, Y)
                del X, Y
            else:
                pool_val = None

            #Multistage model fitting
            model.fit(pool_train, pool_val)
            del pool_train

            #Get best iteration obtained with es
            if pool_val is not None:
                best_iter.append(model.getBestIter())
            else:
                best_iter = -1
            del pool_val


        #Initializing variables
        tot_prauc = 0
        tot_rce = 0
        tot_confmat = [[0,0],[0,0]]
        max_pred = 0 #Max set to the minimum
        min_pred = 1 #Min set to the maximum
        avg = 0
        #Batch evaluation
        for split in tqdm(range(self.tot_test_split)):
            #Iteratively fetching the dataset
            X, Y = get_dataset_xgb_batch(self.tot_test_split, 
                                         split, 
                                         self.test_id, 
                                         self.x_label, 
                                         self.y_label)

            pool_test = self.getPool(X, Y) #------------------------------------- DMATRIX GENERATOR
            del X, Y
            #Multistage evaluation
            prauc, rce, confmat, max_tmp, min_tmp, avg_tmp= model.evaluate(pool_test)
            del pool_test

            #Summing all the evaluations
            tot_prauc = tot_prauc + prauc
            tot_rce = tot_rce + rce
            
            #Computing some statistics for the log
            if self.make_log is True:
                # Getting maximum over iteration
                if max_tmp > max_pred:
                    max_pred = max_tmp
                # Getting minimum over iteration
                if min_tmp < min_pred:
                    min_pred = min_tmp
                # Getting average over itaration
                avg += avg_tmp
                # Computing confusion matrix
                tot_confmat = tot_confmat + confmat
        del model          

        #Averaging the evaluations over # of validation splits
        tot_prauc = tot_prauc/self.tot_test_split
        tot_rce = tot_rce/self.tot_test_split
        avg = avg/self.tot_test_split

        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(best_iter,
                         tot_prauc, 
                         tot_rce, 
                         tot_confmat, 
                         max_pred, 
                         min_pred, 
                         avg,
                         start_log_time)
        
        #Returning the dumbly combined scores
        return self.metriComb(tot_prauc, tot_rce)

    
    # Score function for the CatBoost model
    def blackBoxCatNCV(self, param):
        #Log time
        start_log_time = time.time()
        #print(param)
        #Saving parameters
        if self.make_log is True:
            self.saveParam(param)
        #Initializing the model it it wasn't already
        model = CatBoost(kind = self.kind,
                         verbose=self.verbosity,
                         #Not in tuning dict
                         boosting_type = self.boosting_type,
                         model_shrink_mode = self.model_shrink_mode,
                         leaf_estimation_method = self.leaf_estimation_method,
                         bootstrap_type = self.bootstrap_type,
                         #In tuning dict
                         iterations=                 param[0],
                         depth=                      param[1],
                         learning_rate=              param[2],
                         l2_leaf_reg=                param[3],
                         subsample=                  param[4],
                         random_strenght=            param[5],
                         colsample_bylevel=          param[6],
                         leaf_estimation_iterations= param[7],
                         #scale_pos_weight =          param[8],
                         model_shrink_rate =         param[8],
                         # ES parameters
                         early_stopping_rounds = self.early_stopping_rounds)

        #Iterable returns pair of train - val sets
        id_pairs = zip(TRAIN_IDS, TEST_IDS)

        #Initializing variables
        weight_factor = 0
        averaging_factor = 0
        tot_prauc = 0
        tot_rce = 0
        tot_confmat = [[0,0],[0,0]]
        max_pred = 0
        min_pred = 1
        avg = 0
        best_iter = []
        #Making iterative train-validations
        for dataset_ids in id_pairs:
            weight_factor += weight_factor+1
            averaging_factor += weight_factor
            #Fetching train set
            X, Y = get_dataset_xgb(dataset_ids[0], 
                                   self.x_label, 
                                   self.y_label)
            
            pool_train = self.getPool(X, Y) #------------------------------------- DMATRIX GENERATOR
            del X, Y
            # If early stopping is true for ncv fetching validation set
            # by splitting in two the test set with batch method
            if self.es_ncv is True:
                #Fetching val set 
                X, Y = get_dataset_xgb_batch(2, 0,dataset_ids[1], 
                                             self.x_label, 
                                             self.y_label)
                pool_val = self.getPool(X, Y) #------------------------------------- DMATRIX GENERATOR
                del X, Y
            else:
                pool_val = None
            
            #Multistage model fitting
            model.fit(pool_train, pool_val)
            del pool_train
            if pool_val is not None:
                best_iter.append(model.getBestIter())
            else:
                best_iter = -1
            del pool_val

            if self.es_ncv is True:
                #Fetching test set 
                X, Y = get_dataset_xgb_batch(2, 1,dataset_ids[1], 
                                             self.x_label, 
                                             self.y_label)
                pool_test = self.getPool(X, Y) #------------------------------------- DMATRIX GENERATOR
                del X, Y    
            else:
                X, Y = get_dataset_xgb(dataset_ids[1], 
                                       self.x_label,
                                       self.y_label)
                pool_test = self.getPool(X, Y) #------------------------------------- DMATRIX GENERATOR
                del X, Y
            #Multistage evaluation
            prauc, rce, confmat, max_tmp, min_tmp, avg_tmp= model.evaluate(pool_test)
            del pool_test

            #Weighting scores (based on how many days are in the train set)
            tot_prauc += prauc * weight_factor
            tot_rce += rce * weight_factor

            #Computing some statistics for the log
            if self.make_log is True:
                #Getting maximum over iteration
                if max_tmp > max_pred:
                    max_pred = max_tmp
                #Getting minimum over iteration
                if min_tmp < min_pred:
                    min_pred = min_tmp
                #Getting average over itaration
                avg += avg_tmp
                #Computing the confusion matrix
                tot_confmat = tot_confmat + confmat
        del model

        #Averaging scores
        tot_prauc /= averaging_factor
        tot_rce /= averaging_factor
        #Averaging average (lol)
        avg /= len(TRAIN_IDS)

        #Make human readable logs here
        if self.make_log is True:
            self.saveRes(best_iter,
                         tot_prauc, 
                         tot_rce, 
                         tot_confmat, 
                         max_pred, 
                         min_pred, 
                         avg,
                         start_log_time)
        
        #Returning the dumbly combined scores
        return self.metriComb(tot_prauc, tot_rce)
#------------------------------------------


#-----------------------------------------------------
#           Parameters informations
#-----------------------------------------------------
    # Returns the ordered parameter dictionary
    def getParams(self):
        # Returning an array containing the hyperparameters
        if self.model_name in "xgboost_classifier":
            param_dict = xgbRange(self.kind)

        if self.model_name == "lightgbm_classifier":
            param_dict =  lgbmRange(self.kind)

        if self.model_name == "lightgbm_classifier_cold":
            param_dict =  lgbmRangeCold(self.kind)

        if self.model_name in "catboost_classifier":
            param_dict = catRange(self.kind)

        return param_dict


    # Returns the ordered names parameter dictionary
    def getParamNames(self):
        # Returning the names of the hyperparameters
        if self.model_name in "xgboost_classifier":
            names_dict = xgbName()

        if self.model_name == "lightgbm_classifier":
            names_dict =  lgbmName()

        if self.model_name == "lightgbm_classifier_cold":
            names_dict =  lgbmNameCold()

        if self.model_name in "catboost_classifier":
            names_dict = catName()

        return names_dict
#-----------------------------------------------------


#--------------------------------------------------------------
#               Return the method to optimize
#--------------------------------------------------------------
    # This method returns the score function based on model name
    def getScoreFunc(self):
        if self.mode == 0:
            if self.model_name in "xgboost_classifier":
                score_func = self.blackBoxXGB
        
            if self.model_name == "lightgbm_classifier":
                score_func = self.blackBoxLGB

            if self.model_name == "lightgbm_classifier_cold":
                score_func = self.blackBoxLGBCold

            if self.model_name in "catboost_classifier":
                score_func = self.blackBoxCAT
        elif self.mode == 1:
            if self.model_name in "xgboost_classifier":
                score_func = self.blackBoxXgbBatch

            if self.model_name in "lightgbm_classifier":
                score_func = self.blackBoxLgbBatch

            if self.model_name in "catboost_classifier":
                score_func = self.blackBoxCatBatch
        elif self.mode == 2:
            if self.model_name in "xgboost_classifier":
                score_func = self.blackBoxXgbNCV

            if self.model_name in "lightgbm_classifier":
                score_func = self.blackBoxLgbNCV

            if self.model_name in "catboost_classifier":
                score_func = self.blackBoxCatNCV
        else:
            if self.model_name in "xgboost_classifier":
                score_func = self.blackBoxXgbBatchExtMem

        return score_func
#---------------------------------------------------------------


#-------------------------------------------------   
#           Combine the two metrics
#-------------------------------------------------
    # Returns a combination of the two metrics
    def metriComb(self, prauc, rce):
        # Superdumb combination
        metric = -rce
        if rce > 0:
            metric = - (rce * prauc)
        else:
            metric = - (rce * (1 - prauc))
        if np.isfinite(metric):
            return metric
        else:
            return float(1000)
#-------------------------------------------------


#-------------------------------------------------
#           Load dataset methods
#-------------------------------------------------
    # Loads a custom train set
    def loadTrainData(self, X_train=None, Y_train=None, holder_train=None):
        self.X_train=X_train
        self.Y_train=Y_train
        if holder_train is None:
            if self.model_name in "xgboost_classifier":
                self.train = self.getDMat(X_train, Y_train)

            elif self.model_name in "catboost_classifier":
                self.train = self.getPool(X_train, Y_train)
            '''
            #ADD IF WANT TO USE Dataset FROM LIGHTGBM
            elif self.model_name in "lightgbm_classifier":
                self.train = self.getDataset(X_train, Y_train)
            '''
        else:
            self.train = holder_train

    
    # Loads a custom data set
    def loadValData(self, X_val=None, Y_val=None, holder_val=None):
        self.X_val=X_val
        self.Y_val=Y_val
        if holder_val is None:
            if self.model_name in "xgboost_classifier":
                self.val = self.getDMat(X_val, Y_val)

            elif self.model_name in "catboost_classifier":
                self.val = self.getPool(X_val, Y_val)            
        else:
            self.val = holder_val


    # Loads a custom data set
    def loadTestData(self, X_test=None, Y_test=None, holder_test=None):
        self.X_test=X_test
        self.Y_test=Y_test
        if holder_test is None:
            if self.model_name in "xgboost_classifier":
                self.test = self.getDMat(X_test, Y_test)

            elif self.model_name in "catboost_classifier":
                self.test = self.getPool(X_test, Y_test)
            '''
            #ADD IF WANT TO USE Dataset FROM LIGHTGBM
            elif self.model_name in "lightgbm_classifier":
                self.train = self.getDataset(X_test, Y_test)
            '''
        else:
            self.test = holder_test
#--------------------------------------------------


#--------------------------------------------------
#           Batch/NCV methods
#--------------------------------------------------
    # Passing train set id and number of batches (batch only)
    def batchTrain(self, tot_train_split, train_id):
        self.tot_train_split = tot_train_split
        self.train_id = train_id
    # Passing val set id and number of batches (batch only)
    def batchVal(self, val_id):
        self.val_id = val_id
    # Passing val set id and number of batches (batch only)
    def batchTest(self, tot_test_split, test_id):
        self.tot_test_split = tot_test_split
        self.test_id = test_id
    # Setting labels to use in the sets (batch + NCV)
    # es parameter useful only for NCV
    def setLabels(self, x_label, y_label, es_ncv):
        self.x_label = x_label
        self.y_label = y_label
        self.es_ncv = es_ncv
    def setExtMemTrainPaths(self, ext_memory_paths):
        self.ext_memory_train_paths = ext_memory_paths
    def setExtMemValPaths(self, ext_memory_paths):
        self.ext_memory_val_paths = ext_memory_paths
#--------------------------------------------------


#--------------------------------------------------
#            Save human readable logs
#--------------------------------------------------
    #Saves the parameters (called before the training phase)
    def saveParam(self, param):
        if self.path is None:
            #Taking the path provided
            self.path = str(dt.datetime.now().strftime("%m_%d_%H_%M_%S")) + ".log"
        #Get hyperparameter names
        p_names = self.getParamNames()
        #---------------------------------
        telegram_message=""
        #---------------------------------
        #Opening a file and writing into it the logs
        with open(self.path, 'a') as log:
            to_write = "ITERATION NUMBER " + str(self.iter_count) + "\n"
            log.write(to_write)
            telegram_message+=to_write
            telegram_message+="\n"

            for i in range(len(p_names)):
                to_write = f"'{p_names[i]}': {param[i]}, \n"
                log.write(to_write)
                telegram_message += to_write
                telegram_message += "\n"

        telegram_bot_send_update(self.path+"\n"+telegram_message)


    def saveRes(self, best_iter, prauc, rce, confmat, max_arr, min_arr, avg, log_time):
        if self.path is None:
            # Taking the path provided
            self.path = str(dt.datetime.now().strftime("%m_%d_%H_%M_%S")) + ".log"
        # Opening a file and writing into it the logs
        with open(self.path, 'a') as log:
            # Writing the log
            tn, fp, fn, tp = confmat.ravel()
            total = tn + fp + fn + tp
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * ((precision * recall) / (precision + recall))

            obj = self.metriComb(prauc, rce)
            to_write = "-------\n"
            to_write += "EXECUTION TIME: " + str(time.time()-log_time) + "\n"
            to_write += "-------\n"

            to_write += "best_es_iteration: " + str(best_iter) + "\n"

            to_write += "-------\n"

            to_write += "PRAUC = " + str(prauc) + "\n"
            to_write += "RCE   = " + str(rce) + "\n"

            to_write += "-------\n"

            # confusion matrix in percentages
            to_write += "TN %  = {:0.3f}%\n".format((tn / total) * 100)
            to_write += "FP %  = {:0.3f}%\n".format((fp / total) * 100)
            to_write += "FN %  = {:0.3f}%\n".format((fn / total) * 100)
            to_write += "TP %  = {:0.3f}%\n".format((tp / total) * 100)

            to_write += "-------\n"

            # confusion matrix in percentages
            to_write += "TN    = {:n}\n".format(tn)
            to_write += "FP    = {:n}\n".format(fp)
            to_write += "FN    = {:n}\n".format(fn)
            to_write += "TP    = {:n}\n".format(tp)

            to_write += "-------\n"

            to_write += "PREC    = {:0.5f}%\n".format(precision)
            to_write += "RECALL    = {:0.5f}%\n".format(recall)
            to_write += "F1    = {:0.5f}%\n".format(f1)

            to_write += "-------\n"

            to_write += "MAX   =" + str(max_arr) + "\n"
            to_write += "MIN   =" + str(min_arr) + "\n"
            to_write += "AVG   =" + str(avg) + "\n"

            to_write += "-------\n"

            to_write += "OBJECTIVE: " + str(obj) + "\n\n\n"
            log.write(to_write)
            telegram_bot_send_update(self.path+"\n"+to_write)

        # Increasing the iteration count
        self.iter_count = self.iter_count + 1

    #--------------------------------------------------


#--------------------------------------------------
#             Reset saving parameters
#--------------------------------------------------
# Stop appending to the previos log and make a new
# one.
#--------------------------------------------------
    def resetSaveLog(self):
        self.iter_count = 1
        self.path = None
#--------------------------------------------------


#--------------------------------------------------
#             Setting log saver
#--------------------------------------------------
    def setSaveLog(self, make_log):
        self.make_log=make_log
#--------------------------------------------------


#--------------------------------------------------
#            Setting logs path
#--------------------------------------------------
    def setLogPath(self, path):
        self.path = path
#--------------------------------------------------


#--------------------------------------------------
#        Setting non tuned parameters for xgb
#--------------------------------------------------
    def setParamsXGB(self, verbosity, 
                           process_type, 
                           tree_method, 
                           objective, 
                           num_parallel_tree, 
                           eval_metric, 
                           early_stopping_rounds):
        self.verbosity=verbosity
        self.process_type=process_type
        self.tree_method=tree_method
        self.objective=objective
        self.num_parallel_tree=num_parallel_tree
        self.eval_metric=eval_metric
        self.early_stopping_rounds=early_stopping_rounds
#--------------------------------------------------


#--------------------------------------------------
#        Setting non tuned parameters for xgb
#--------------------------------------------------
    def setParamsLGB(self, verbosity, 
                           process_type, 
                           tree_method, 
                           objective, 
                           num_parallel_tree, 
                           eval_metric, 
                           early_stopping_rounds,
                           is_unbalance=False):
        self.verbosity=verbosity
        self.process_type=process_type
        self.tree_method=tree_method
        self.objective=objective
        self.num_parallel_tree=num_parallel_tree
        self.eval_metric=eval_metric
        self.early_stopping_rounds=early_stopping_rounds
        self.is_unbalance=is_unbalance
#--------------------------------------------------

#--------------------------------------------------
#       Setting non tuned parameters for cat
#--------------------------------------------------
    def setParamsCAT(self, verbosity,
                           boosting_type,
                           model_shrink_mode,
                           leaf_estimation_method,
                           bootstrap_type,
                           early_stopping_rounds):
        self.verbosity = verbosity
        self.boosting_type = boosting_type
        self.model_shrink_mode = model_shrink_mode
        self.leaf_estimation_method = leaf_estimation_method
        self.bootstrap_type = bootstrap_type
        self. early_stopping_rounds = early_stopping_rounds
#--------------------------------------------------

#--------------------------------------------------
#           Set the categorical features
#--------------------------------------------------
#              Used in Cat and Lgbm
#--------------------------------------------------
    def setCategoricalFeatures(self,categorical_features=None):
        self.categorical_features = categorical_features

#--------------------------------------------------
#           Generator of DMatrices
#--------------------------------------------------
#              Used in XGBoost
#--------------------------------------------------
    def getDMat(self, X, Y=None):
        return xgb.DMatrix(X, label=Y)
#--------------------------------------------------


#--------------------------------------------------
#             Generator of Train Pool
#--------------------------------------------------
#              Used in CatBoost
#--------------------------------------------------
#    def getTrainPool(self, X, Y=None):
#        #l = np.array(Y).astype(np.int32)
#        X = np.array(X)
#        Y = np.array(Y)
#        print("Creating Pool:")
#        return cat.Pool(X, label=Y, cat_features=self.categorical_features)
#        print("Pool created.")
#--------------------------------------------------

#--------------------------------------------------
#             Generator of Eval Pool
#--------------------------------------------------
#              Used in CatBoost
#--------------------------------------------------
    def getPool(self, X, Y=None):
        X = np.array(X)
        Y = np.array(Y)
        if Y is not None:
            l = np.array(Y).astype(np.int32)
        else:
            l = Y # which is None
        return cat.Pool(X, label=l, cat_features=self.categorical_features)
#--------------------------------------------------

#----------------------------------------------------------
def getDMat(X, Y=None):
    return xgb.DMatrix(X, label=Y)


def run_xgb_external_memory(param, model_interface, queue):
    #Saving parameters of the optimization
    if model_interface.make_log is True:
        model_interface.saveParam(param)
    # Initializing the model it it wasn't already
    model = XGBoost(kind=model_interface.kind,
                    # Not in tuning dict
                    objective="binary:logistic",
                    num_parallel_tree=4,
                    eval_metric="auc",
                    # In tuning dict
                    num_rounds=param[0],
                    max_depth=param[1],
                    min_child_weight=param[2],
                    colsample_bytree=param[3],
                    learning_rate=param[4],
                    reg_alpha=param[5],
                    reg_lambda=param[6],
                    scale_pos_weight=param[7],
                    gamma=param[8],
                    subsample=param[9],
                    base_score=param[10],
                    max_delta_step= param[11])

    # Batch train
    for path in tqdm(model_interface.ext_memory_train_paths):
        # Multistage model fitting
        dmat_train = getDMat(path) #------------------------------------- DMATRIX GENERATOR
        model.fit(dmat_train)
        del dmat_train

    # TODO: last evaluation set may be smaller so it needs
    # to be weighted according to its dimension.

    # Initializing variables
    tot_prauc = 0
    tot_rce = 0
    tot_confmat = [[0, 0], [0, 0]]
    max_pred = 0  # Max set to the minimum
    min_pred = 1  # Min set to the maximum
    avg = 0
    model_interface.tot_val_split = len(model_interface.ext_memory_val_paths)
    # Batch evaluation
    for path in tqdm(model_interface.ext_memory_val_paths):
        # Iteratively fetching the dataset
        dmat_test = xgb.DMatrix(path, silent=True)
        
        # Multistage evaluation
        prauc, rce, confmat, max_tmp, min_tmp, avg_tmp = model.evaluate(dmat_test)
        del dmat_test

        # Summing all the evaluations
        tot_prauc = tot_prauc + prauc
        tot_rce = tot_rce + rce

        # Computing some statistics for the log
        if model_interface.make_log is True:
            # Getting maximum over iteration
            if max_tmp > max_pred:
                max_pred = max_tmp
            # Getting minimum over iteration
            if min_tmp < min_pred:
                min_pred = min_tmp
            # Getting average over itaration
            avg += avg_tmp
            # Computing confusion matrix
            tot_confmat = tot_confmat + confmat
    del model

    # Averaging the evaluations over # of validation splits
    tot_prauc = tot_prauc / model_interface.tot_val_split
    tot_rce = tot_rce / model_interface.tot_val_split
    avg = avg / model_interface.tot_val_split

    # Make human readable logs here
    if model_interface.make_log is True:
        model_interface.saveRes(tot_prauc,
                     tot_rce,
                     tot_confmat,
                     max_pred,
                     min_pred,
                     avg)

    # Returning the dumbly combined scores
    queue.put(model_interface.metriComb(tot_prauc, tot_rce))
    return model_interface.metriComb(tot_prauc, tot_rce)
#--------------------------------------------------------------
