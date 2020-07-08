import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, log_loss
import time
import pickle
import os.path
import datetime as dt
import sys
import math
from Utils.Base.RecommenderGBM import RecommenderGBM
from Utils.Eval.Metrics import ComputeMetrics as CoMe
from Utils.Data import Data
from Utils.Submission.Submission import create_submission_file
from Models.GBM.XGBoostMulti import XGBoostMulti






def main():
    onehot = pd.read_csv("onehot.csv", sep='\x01')
    #u_dict = load_obj("u_dict")
    #i_dict = load_obj("i_dict")
    
    
    #XGBoost part
    test_size = 0.2
    #Dividing the dataset splitting the column i need to predict from the others
    X = onehot[["usr_id", "twt_id"]].to_numpy()
    Y = onehot[["tmstp_lik", "tmstp_rpl", "tmstp_rtw", "tmstp_rtw_c"]].to_numpy()
    XGB = XGBoostMulti(kind="MULTI")

    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=test_size, 
                                                        random_state=int(time.time()))

    XGB.fit(X_train, Y_train)
    #print(XGB.evaluate(X_test, Y_test))
    x = XGB.get_prediction(X_test)
    for i in x:
        print(i)







if __name__ == "__main__":
    main()

