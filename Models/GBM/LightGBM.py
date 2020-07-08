import os.path

import lightgbm as lgb
import numpy as np

from Utils.Base.RecommenderGBM import RecommenderGBM
from Utils.Eval.ConfMatrix import confMatrix
from Utils.Eval.Metrics import ComputeMetrics as CoMe
import pandas as pd
import sklearn.inspection as sk_ins

# TODO: The categorical features must be imported from load_data() method
from Utils.Importance.LGBMImportance import LGBMImportance


class LightGBM(RecommenderGBM):
    # ---------------------------------------------------------------------------------------------------
    # n_rounds:      Number of rounds for boosting
    # param:         Parameters of the XGB model
    # kind:          Name of the kind of prediction to print [LIKE, REPLY, REWTEET, RETWEET WITH COMMENT]
    # ---------------------------------------------------------------------------------------------------
    # Not all the parameters are explicitated
    # PARAMETERS DOCUMENTATION:  https://lightgbm.readthedocs.io/en/latest/Parameters.html
    # ---------------------------------------------------------------------------------------------------
    def __init__(self,
                 kind="NO_NAME_GIVEN",
                 batch=False,
                 # Not in tuning dict
                 objective='binary',
                 metric='binary',
                 num_threads=-1,
                 # In tuning dict
                 num_iterations=1000,
                 num_leaves=31,
                 learning_rate=0.2,
                 max_depth=14,
                 lambda_l1=0.01,
                 lambda_l2=0.01,
                 colsample_bynode=0.5,
                 colsample_bytree=0.5,
                 pos_subsample=0.5,  # In classification positive and negative-
                 neg_subsample=0.5,  # Subsample ratio.
                 scale_pos_weight=None,  # (same as xgboost)
                 is_unbalance=False,
                 # let LightGBM deal with the unbalance problem on its own (worsen RCE 100x in local if used alone)
                 bagging_freq=5,  # Default 0 perform bagging every k iterations
                 bagging_fraction=1,  # Rnadomly selects specified fraciton of the data without resampling
                 min_data_in_leaf=20,
                 max_bin=255,
                 # For early stopping
                 early_stopping_rounds=None):

        super(LightGBM, self).__init__(
            name="lightgbm_classifier",  # name of the recommender
            kind=kind,  # what does it recommends
            batch=batch)  # if it's a batch type

        # INPUTS
        self.kind = kind
        self.batch = batch

        self.params = {
            # Parameters
            'objective': objective,
            'metric': metric,
            'num_iterations': num_iterations,
            'num_leaves': int(num_leaves),
            'learning_rate': learning_rate,
            'num_threads': num_threads,
            'max_depth': max_depth,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'colsample_bytree': colsample_bytree,
            'colsample_bynode': colsample_bynode,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'max_bin': int(max_bin),
            'boost_from_average': True,  # default: True
            # 'boosting': "dart"                     #default: gbdt, it is said that dart provides more acurate predictions, while risking overfitting tho
            'early_stopping_round': early_stopping_rounds,
            'is_unbalance': is_unbalance,
            'min_data_in_leaf': int(min_data_in_leaf),
            'bagging_seed': 92492
        }

        if scale_pos_weight is None and is_unbalance is None:
            self._initStandard(self.params)
        elif scale_pos_weight is None:
            self._initUnbalance(self.params)
        elif is_unbalance is None:
            self._initScalePosWeight(self.params)
        else:
            print(
                "[WARNING] Errors in the configuration. scale_pos_weight and is_unbalance can't both be True at the same time.")
            print("Starting default configuration.")
            self._initStandard(self.params)

        # CLASS VARIABLES
        # Models
        self.model = None
        # Default for categorical features
        self.categorical_feature = 'auto'

        # Extension of saving file
        self.ext = ".txt"

    def _initStandard(self, parameters):
        # PARAMETERS' RANGE DICTIONARY
        self.param_range_dict = {'num_iterations': (0, 500),
                                 'num_rounds': (5, 1000),
                                 'num_leaves': (15, 500),
                                 'learning_rate': (0.00001, 0.1),
                                 'max_depth': (15, 500),
                                 'lambda_l1': (0.00001, 0.1),
                                 'lambda_l2': (0.00001, 0.1),
                                 'colsample_bynode': (0, 1),
                                 'colsample_bytree': (0, 1),
                                 'subsample': (0, 1),
                                 'pos_subsample': (0, 1),
                                 # with these two parameters we can regulate the unbalanced problems
                                 'neg_subsample': (0, 1),  # so we may set the ranges differently for every problem
                                 'bagging_freq': (0, 1),
                                 'bagging_fraction': (0, 1)}

    def _initUnbalance(self, parameters):
        # PARAMETERS' RANGE DICTIONARY
        self.param_range_dict = {'num_iterations': (0, 500),
                                 'num_rounds': (5, 1000),
                                 'num_leaves': (15, 500),
                                 'learning_rate': (0.00001, 0.1),
                                 'max_depth': (15, 500),
                                 'lambda_l1': (0.00001, 0.1),
                                 'lambda_l2': (0.00001, 0.1),
                                 'colsample_bynode': (0, 1),
                                 'colsample_bytree': (0, 1),
                                 'subsample': (0, 1),
                                 'pos_subsample': (0, 1),
                                 # with these two parameters we can regulate the unbalanced problems
                                 'neg_subsample': (0, 1),  # so we may set the ranges differently for every problem
                                 'bagging_freq': (0, 1),
                                 'bagging_fraction': (0, 1)}

    def _initScalePosWeight(self, parameters):
        # PARAMETERS' RANGE DICTIONARY
        self.param_range_dict = {'num_iterations': (0, 500),
                                 'num_rounds': (5, 1000),
                                 'num_leaves': (15, 500),
                                 'learning_rate': (0.00001, 0.1),
                                 'max_depth': (15, 500),
                                 'lambda_l1': (0.00001, 0.1),
                                 'lambda_l2': (0.00001, 0.1),
                                 'colsample_bynode': (0, 1),
                                 'colsample_bytree': (0, 1),
                                 'subsample': (0, 1),
                                 'bagging_freq': (0, 1),
                                 'bagging_fraction': (0, 1)}

    def fit(self, X=None, Y=None, X_val=None, Y_val=None, categorical_feature=None):
        # Tries to load X and Y if not directly passed
        if (X is None) or (Y is None):
            X, Y = self.load_data(self.test_dataset)  # still to be implemented
            print("Train set loaded from file.")

        # If categorical features are null taking default value
        if (categorical_feature is None):
            categorical_feature = self.categorical_feature

        # Declaring LightGBM Dataset
        train = lgb.Dataset(data=X,
                            label=Y,
                            categorical_feature=categorical_feature)
        del X, Y

        # Learning in a single round
        if self.model is None:
            if X_val is None:

                # Defining and fitting the model
                self.model = lgb.train(self.get_param_dict(),
                                       train_set=train,
                                       categorical_feature=categorical_feature)
            else:

                validation = lgb.Dataset(data=X_val, label=Y_val, categorical_feature=categorical_feature)
                del X_val, Y_val

                # Defining and fitting the model
                self.model = lgb.train(self.get_param_dict(),
                                       train_set=train,
                                       valid_sets=validation,
                                       categorical_feature=categorical_feature)

        # Learning by consecutive batches
        else:
            # Defining and training the models
            # self.model = lgb.train(self.get_param_dict(),
            #                              train_set=train,
            #                              init_model=self.model,
            #                              categorical_feature=categorical_feature)
            raise NotImplementedError("Batch training is not implemented in lgbm yet.")

    # Returns the predictions and evaluates them
    # ---------------------------------------------------------------------------
    #                           evaluate(...)
    # ---------------------------------------------------------------------------
    # X_tst:     Features of the test set
    # Y_tst      Ground truth, target of the test set
    # ---------------------------------------------------------------------------
    #           Works for both for batch and single training
    # ---------------------------------------------------------------------------
    def evaluate(self, X_tst=None, Y_tst=None):
        Y_pred = None

        # Tries to load X and Y if not directly passed
        if (X_tst is None) or (Y_tst is None):
            X_tst, Y_tst = Data.get_dataset_xgb_default_test()
            print("Test set loaded from file.")
        # Y_tst = np.array(Y_tst[Y_tst.columns[0]].astype(float))

        if self.model is None:
            print("No model trained yet.")
        else:
            # Selecting the coherent model for the evaluation
            # According to the initial declaration (batch/single round)
            # model = self.get_model()

            # Preparing DMatrix
            # d_test = xgb.DMatrix(X_tst)
            # Making predictions
            # Y_pred = model.predict(d_test)
            Y_pred = self.get_prediction(X_tst)

            # Declaring the class containing the
            # metrics.
            cm = CoMe(Y_pred, Y_tst)

            # Evaluating
            prauc = cm.compute_prauc()
            rce = cm.compute_rce()
            # Confusion matrix
            conf = confMatrix(Y_tst, Y_pred)
            # Prediction stats
            max_pred = max(Y_pred)
            min_pred = min(Y_pred)
            avg = np.mean(Y_pred)

            # mancano da ritornare best_iter, nax_arr e min_arr
            return prauc, rce, conf, max_pred, min_pred, avg

    # This method returns only the predictions
    # -------------------------------------------
    #           get_predictions(...)
    # -------------------------------------------
    # X_tst:     Features of the test set
    # -------------------------------------------
    # As above, but without computing the scores
    # -------------------------------------------
    def get_prediction(self, X_tst=None):
        Y_pred = None
        # Tries to load X and Y if not directly passed
        if (X_tst is None):
            X_tst, Y_tst = self.load_data(self.test_dataset)
            print("Test set loaded from file.")
        if self.model is None:
            print("No model trained yet.")
        else:
            # Selecting the coherent model for the evaluation
            model = self.model

            # Making predictions
            # Wants row data for prediction
            Y_pred = model.predict(X_tst)
            return Y_pred

    # This method loads a model
    # -------------------------
    # path: path to the model
    # -------------------------
    def load_model(self, path):
        # Reinitializing model
        self.model = lgb.Booster(model_file=path)
        print("Model correctly loaded.\n")

    def save_model(self, path=None, filename=None):
        self.model.save_model(filename)
        print(f"Model correctly saved in {filename}.\n")

    # Returns/prints the importance of the features
    # -------------------------------------------------
    # verbose:   it also prints the features importance
    # -------------------------------------------------
    def get_feat_importance(self, verbose=False):

        model = self.model

        # Getting the importance
        importance = model.feature_importance(importance_type="gain")

        if verbose is True:
            print("F_pos\tF_importance")
            for k in range(len(importance)):
                print("{0}:\t{1}".format(k, importance[k]))

        return importance

    def plot_fimportance(self):
        model = self.model
        import matplotlib.pyplot as plt

        lgb.plot_importance(model, importance_type="split")
        plt.rcParams["figure.figsize"] = (50, 50)
        plt.title("Feature Importance: SPLIT")
        plt.savefig("fimportance_split.png")
        lgb.plot_importance(model, importance_type="gain")
        plt.rcParams["figure.figsize"] = (20, 5)
        plt.title("Feature Importance: GAIN")
        plt.savefig("fimportance_gain.png")

    def save_fimportance_df(self, filename: str):
        model = self.model

        feature_imp_split = pd.DataFrame({'Value': model.feature_importance(importance_type="split"),
                                          'Feature': model.feature_name()})
        feature_imp_split.to_csv(filename + '_split.csv')

        feature_imp_gain = pd.DataFrame(
            {'Value': model.feature_importance(importance_type="gain"), 'Feature': model.feature_name()})
        feature_imp_gain.to_csv(filename + '_gain.csv')

    # Returns the parameters in dictionary form
    def get_param_dict(self):
        return self.params

    # This method loads the dataset from file
    # -----------------------------------------------
    # dataset:   defines if will be load the train or
    #           test set, should be equal to either:
    #
    #           self.train_dataset
    #           self.test_dataset
    # -----------------------------------------------
    # TODO: has to be extended if we want to use it
    # too for LightGBM and CatBoost
    # -----------------------------------------------
    def load_data(self, dataset):
        # X, Y = Data.get_dataset_xgb(train_dataset, X_label, Y_label)
        # X, Y, self.categorical_feature = Data.get_dataset_xgb(...)
        X = None
        Y = None
        return X, Y

    def get_model(self):
        # Selecting the coherent model for the evaluation
        # According to the initial declaration (batch/single round)
        return self.model

    def get_best_iter(self):
        return self.model.best_iteration

    def permutation_importance(self, X_tst, Y_tst):
        lgbm_importance = LGBMImportance(model=self)
        r = sk_ins.permutation_importance(lgbm_importance, X_tst, Y_tst, n_jobs=8)
        outfile = open("lgbm_permutation_importance.csv", "a+")
        for i in r.importances_mean.argsort()[::-1]:
            print(f"{X_tst.columns[i]},"
                  f"{r.importances_mean[i]:.3f},"
                  f"{r.importances_std[i]:.3f}")
            outfile.write(f"{X_tst.columns[i]},"
                          f"{r.importances_mean[i]:.3f},"
                          f"{r.importances_std[i]:.3f}\n")
