import math
import pickle

import xgboost as xgb

from Utils.Base.RecommenderGBM import RecommenderGBM
from Utils.Eval.Metrics import ComputeMetrics as CoMe, CustomEvalXGBoost


class XGBoost(RecommenderGBM):
    # ---------------------------------------------------------------------------------------------------
    # n_rounds:      Number of rounds for boosting
    # param:         Parameters of the XGB model
    # kind:          Name of the kind of prediction to print [LIKE, REPLY, REWTEET, RETWEET WITH COMMENT]
    # ---------------------------------------------------------------------------------------------------
    # Not all the parameters are explicitated
    # PARAMETERS DOCUMENTATION:https://xgboost.readthedocs.io/en/latest/parameter.html
    # ---------------------------------------------------------------------------------------------------

    def __init__(self,
                 kind="NO_KIND_GIVEN",
                 # Not in tuning dict
                 verbosity=2,
                 process_type="default",
                 tree_method="hist",
                 objective="binary:logistic",  # outputs the binary classification probability
                 num_parallel_tree=4,  # Number of parallel trees
                 eval_metric="rmsle",  # WORKS ONLY IF A VALIDATION SET IS PASSED IN TRAINING PHASE
                 early_stopping_rounds=None,
                 # In tuning dict
                 num_rounds=10,
                 colsample_bytree=1,
                 learning_rate=0.3,
                 max_depth=6,  # Max depth per tree
                 reg_alpha=0,  # L1 regularization
                 reg_lambda=1,  # L2 regularization
                 min_child_weight=1,  # Minimum sum of instance weight (hessian) needed in a child.
                 scale_pos_weight=1,
                 gamma=0,
                 max_delta_step= 0,
                 base_score=0.5,
                 subsample=1):

        super(XGBoost, self).__init__(
            name="xgboost_classifier",  # name of the recommender
            kind=kind)  # what does it recommends
            

        # INPUTS
        self.kind = kind
        # Parameters
        # Not in dict
        self.verbosity = verbosity
        self.process_type = process_type
        self.tree_method = tree_method
        self.objective = objective
        self.num_parallel_tree = num_parallel_tree
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        # In dict
        self.num_rounds = num_rounds
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.num_parallel_tree = num_parallel_tree
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.subsample = subsample
        self.gamma = gamma
        self.max_delta_step=max_delta_step
        self.base_score = base_score
        self.subsample = subsample

        # CLASS VARIABLES
        # Model
        self.model = None
        # Prediction
        self.Y_pred = None
        # Extension of saved file
        self.ext = ".model"


    # -----------------------------------------------------
    #                    fit(...)
    # -----------------------------------------------------
    # dmat_train:  Training set in DMatrix form.
    # dmat_val:    Validation set in DMatrix form provided
    #              in order to use Early Stopping.
    # -----------------------------------------------------
    # sround_model and batch_model are differentiated
    # in order to avoid overwriting. (Maybe not necessary)
    # -----------------------------------------------------
    # TODO: Redoundant code here
    #------------------------------------------------------
    def fit(self, dmat_train=None, dmat_val=None):
        # In case validation set is not provided set early stopping rounds to default
        if (dmat_val is None):
            self.early_stopping_rounds = None
            dmat_val = []
        else:
            dmat_val = [(dmat_val, "eval")]

        custom_eval_xgb = CustomEvalXGBoost()

        if self.model is not None:
            # Continue the training og a model already saved
            self.model = xgb.train(self.get_param_dict(),
                                   num_boost_round=math.ceil(self.num_rounds),
                                   early_stopping_rounds=self.early_stopping_rounds,
                                   evals=dmat_val,
                                   dtrain=dmat_train,
                                   #feval=custom_eval_xgb.custom_eval,
                                   xgb_model=self.model)

        # if we have no model saved
        else:
            self.model = xgb.train(self.get_param_dict(),
                                   num_boost_round=math.ceil(self.num_rounds),
                                   early_stopping_rounds=self.early_stopping_rounds,
                                   evals=dmat_val,
                                   #feval=custom_eval_xgb.custom_eval,
                                   dtrain=dmat_train)


    # Returns the predictions and evaluates them
    # ---------------------------------------------------------------------------
    #                           evaluate(...)
    # ---------------------------------------------------------------------------
    # X_tst:     Features of the test set
    # Y_tst      Ground truth, target of the test set
    # ---------------------------------------------------------------------------
    #           Works for both for batch and single training
    # ---------------------------------------------------------------------------
    def evaluate(self, dmat_test=None):
        # Tries to load X and Y if not directly passed
        if (dmat_test is None):
            print("No matrix passed, cannot perform evaluation.")
        
        if (self.model is None):
            print("No model trained, cannot to perform evaluation.")

        else:
            #Retrieving the predictions
            Y_pred = self.get_prediction(dmat_test)

            # Declaring the class containing the metrics
            cm = CoMe(Y_pred, dmat_test.get_label())

            # Evaluating
            prauc = cm.compute_prauc()
            rce = cm.compute_rce()
            # Confusion matrix
            conf = cm.confMatrix()
            # Prediction stats
            max_pred, min_pred, avg = cm.computeStatistics()

            return prauc, rce, conf, max_pred, min_pred, avg


    # This method returns only the predictions
    # -------------------------------------------
    #           get_predictions(...)
    # -------------------------------------------
    # X_tst:     Features of the test set
    # -------------------------------------------
    # As above, but without computing the scores
    # -------------------------------------------
    def get_prediction(self, dmat_test=None):
        # Tries to load X and Y if not directly passed
        if (dmat_test is None):
            print("No matrix passed, cannot provide predictions.")

        if (self.model is None):
            print("No model trained, cannot perform evaluation.")

        else:
            # Making predictions
            Y_pred = self.model.predict(dmat_test)
            return Y_pred


    #--------------------------
    # This method loads a model
    # -------------------------
    # path: path to the model
    # -------------------------
    def load_model(self, path):
        self.model = pickle.load(open(f"{path}", "rb"))


    #--------------------------------------------------
    # Returns/prints the importance of the features
    # -------------------------------------------------
    # verbose:   it also prints the features importance
    # -------------------------------------------------
    def get_feat_importance(self, verbose=False):
        
        importance = self.model.get_score(importance_type='gain')        

        if verbose is True:
            for k, v in importance.items():
                print("{0}:\t{1}".format(k, v))

        return importance


    # Returns parameters in dicrionary form
    def get_param_dict(self):
        param_dict = {'verbosity': self.verbosity,
                      'process_type': self.process_type,
                      'tree_method': self.tree_method,
                      'objective': self.objective,
                      'eval_metric': self.eval_metric,
                      #'disable_default_eval_metric': 1,
                      'colsample_bytree': self.colsample_bytree,
                      'learning_rate': self.learning_rate,
                      'max_depth': math.ceil(self.max_depth),
                      'reg_alpha': self.reg_alpha,
                      'reg_lambda': self.reg_lambda,
                      'num_parallel_tree': self.num_parallel_tree,
                      'min_child_weight': self.min_child_weight,
                      'scale_pos_weight': self.scale_pos_weight,
                      'subsample': self.subsample,
                      'gamma': self.gamma,
                      'max_delta_step':self.max_delta_step,
                      'base_score': self.base_score
                     }

        return param_dict


    
    #-----------------------------------------------------
    #        Get the best iteration with ES
    #-----------------------------------------------------
    def getBestIter(self):
        return self.model.best_iteration


