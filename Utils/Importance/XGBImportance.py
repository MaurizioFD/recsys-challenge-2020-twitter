import os

import xgboost as xgb

from Models.GBM.XGBoost import XGBoost
from Utils.Data.DataUtils import cache_dataset_as_svm
from Utils.Eval.Metrics import ComputeMetrics


class XGBImportance:

    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, path):
        model = XGBoost()
        model.load_model(path)
        return model

    def fit(self, *params):
        print("FIT PARAMS")
        print(params)

    def score(self, X_test, Y_test):
        # cache_dataset_as_svm(f"/home/ubuntu/data/rec_sys_challenge_2020/perm_importance/remote_val", X_test, Y_test, no_fuck_my_self=True)
        X_test = xgb.DMatrix(X_test, missing=0, silent=False)
        predictions = self.model.get_prediction(dmat_test=X_test)
        cm = ComputeMetrics(predictions, Y_test.to_numpy())
        # Evaluating
        rce = cm.compute_rce()

        print(rce)
        return rce
