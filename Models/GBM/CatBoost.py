import numpy as np
from catboost import CatBoostClassifier

from Utils.Base.RecommenderGBM import RecommenderGBM
from Utils.Eval.Metrics import ComputeMetrics as CoMe

#-----------------------------------------------------------------------------
#                           CATBOOST CLASSIFIER
#-----------------------------------------------------------------------------
#                            About Parameters
#-----------------------------------------------------------------------------
# loss_function:     Metric to use in train.
# eval_metric:       Metric used for overfitting detection (Early
#                     stopping). Can be custom. Check the link:
'''
https://catboost.ai/docs/concepts/python-usages-examples.html#custom-loss-function-eval-metric
'''
# iterations:        The maximum number of trees that can be built.
# depth:         
# learning_rate:     The learning rate.
# l2_leaf_reg:       Coefficient at the L2 regularization term.
# booststrap_type:   Method for sampling the weights of objects.
# subsample:         Sample rate for bagging.
# random_strenght:   Amount of randomness to use for scoring splits
#                     when the tree structure is selected. Use this
#                     paraeter to avoid overfitting.
# depth:             Depth of the tree (max. to 16 for CPU).
# min_data_in_leaf:  The minimum number of training sample in a leaf.
# max_leaves:        Maximum number of leaves in resulting tree.
# colsample_bylevel: Random subspace method. Percentage of features
#                     to use at each split selection.
# leaf_estimation_method:   Method used to calculate values in leaves.
#                            (Newton good for classification).
# leaf_estimation_iterations: Number of gradient steps when calculating
#                              the values in leaves.
# scale_pos_weight:           Weight for class 1 in binary classification
#                              (the probability of a record to be positive).
# boosting_type:              Plain value for big datasets. (Clasisg gradient
#                              boosting).
# model_shrink_rate:          The constant used to calculate teh cohefficient
#                              for multiplying the model on each iteration.
# model_shrink_mode           Determines how the actual model shrinkage
#                              cohefficient is calculated at each iteration.
#-----------------------------------------------------------------------------
# TODO: ADD PARAMETERS FOR CATEGORICAL FEATURES
#-----------------------------------------------------------------------------
# For the complete list of parameters check this link:
# https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list
#-----------------------------------------------------------------------------
# Check this for random_strenght and bagging_temperature:
# https://github.com/catboost/catboost/issues/373
#-----------------------------------------------------------------------------

class CatBoost(RecommenderGBM):
    def __init__(self,
                 kind="NO_NAME_GIVEN",
                 #Not in tuning dict
                 thread_count=-1,
                 verbose=True,
                 loss_function="Logloss",
                 eval_metric="Logloss",
                 #In tuning dict
                 iterations=20,
                 depth=16,
                 learning_rate = 0.1,
                 l2_leaf_reg = 0.01,
                 bootstrap_type = "Bernoulli",
                 subsample = 0.8,
                 random_strenght = 0.5,
                 max_leaves = 31,
                 colsample_bylevel = 0.5,
                 leaf_estimation_method = "Newton",
                 leaf_estimation_iterations= 10,
                 scale_pos_weight = 1,
                 boosting_type = "Plain",
                 model_shrink_mode = "Constant",
                 model_shrink_rate = 0.5,
                 early_stopping_rounds = 10,
                 od_type = "Iter"): #Uses early stopping instead of another method

        super(CatBoost, self).__init__(
              name="catboost_classifier",
              kind=kind)

        #Inputs
        #self.param=param
        self.kind=kind

        #TODO: Dictionary containing pamarameters' range
        self.param_dict = None

        #CLASS VARIABLES
        #Model
        self.model = None
        #Save extension
        self.ext=".cbm"
        #Cannot pass parameters as a dict
        #Explicitating parameters (set to default)
        self.loss_function=loss_function
        self.eval_metric=eval_metric
        self.verbose=verbose
        self.iterations=iterations
        self.depth=depth
        self.learning_rate=learning_rate
        self.l2_leaf_reg=l2_leaf_reg
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.max_leaves = max_leaves
        self.leaf_estimation_method = leaf_estimation_method
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.scale_pos_weight = scale_pos_weight
        self.boosting_type = boosting_type
        self.model_shrink_mode = model_shrink_mode
        self.model_shrink_rate = model_shrink_rate
        self.random_strenght = random_strenght
        self.colsample_bylevel = colsample_bylevel
        # ES parameters
        self.early_stopping_rounds = early_stopping_rounds
        self.od_type = od_type
        # Number of threads
        self.thread_count = thread_count


    def init_model(self):
        return CatBoostClassifier(loss_function= self.loss_function,
                                  eval_metric= self.eval_metric,
                                  verbose= self.verbose,
                                  iterations= self.iterations,
                                  depth= self.depth,
                                  learning_rate= self.learning_rate,
                                  l2_leaf_reg= self.l2_leaf_reg,
                                  bootstrap_type=self.bootstrap_type,
                                  subsample=self.subsample,
                                  max_leaves=self.max_leaves,
                                  leaf_estimation_method=self.leaf_estimation_method,
                                  leaf_estimation_iterations=self.leaf_estimation_iterations,
                                  scale_pos_weight=self.scale_pos_weight,
                                  boosting_type=self.boosting_type,
                                  model_shrink_mode=self.model_shrink_mode,
                                  model_shrink_rate=self.model_shrink_rate,
                                  random_strength=self.random_strenght,
                                  colsample_bylevel=self.colsample_bylevel,
                                  od_wait=self.early_stopping_rounds,       # ES set here
                                  od_type=self.od_type,                     # ES set here
                                  thread_count=self.thread_count)
        

    #-----------------------------------------------------
    #                    fit(...)
    #-----------------------------------------------------
    # pool_train:   Pool containing the training set.
    # pool_val:     Pool contaning the validation set.
    # cat_feat:     Array containing the indices of the
    #                columns containing categorical feat.
    #-----------------------------------------------------
    def fit(self, pool_train = None, pool_val = None, cat_feat=None):

        # In case validation set is not provided set early stopping rounds to default
        if (pool_val is None):
            self.early_stopping_rounds = None
            self.od_type = None

        if pool_train is None:
            print("No train set provided.")

        elif self.model is not None:
            # Fitting again an already trained model
            self.model.fit(pool_train,
                           cat_features=cat_feat,
                           eval_set=pool_val,
                           init_model=self.model)

        else:
            # Defining and fitting the model
            self.model = self.init_model()
            self.model.fit(pool_train,
                           cat_features=cat_feat,
                           eval_set=pool_val)



    # Returns the predictions and evaluates them
    #---------------------------------------------------------------------------
    #                           evaluate(...)
    #---------------------------------------------------------------------------
    #X_tst:     Features of the test set
    #Y_tst      Ground truth, target of the test set
    #---------------------------------------------------------------------------
    #           Works for both for batch and single training
    #---------------------------------------------------------------------------
    def evaluate(self, pool_tst=None):
        if (pool_tst is None):
            print("No dataset provided.")    
        if (self.model is None):
            print("No model trained yet.")
        else:            
            #Preparing DMatrix
            #p_test = Pool(X_tst, label=Y_tst)
            #Making predictions
            #Y_pred = model.predict_proba(p_test)
            Y_pred = self.get_prediction(pool_tst)

            # Declaring the class containing the
            # metrics.
            Y_test = np.array(pool_tst.get_label()).astype(np.int32)
            cm = CoMe(Y_pred, Y_test)

            # Evaluating
            prauc = cm.compute_prauc()
            rce = cm.compute_rce()
            # Confusion matrix
            conf = cm.confMatrix()
            # Prediction stats
            max_pred, min_pred, avg = cm.computeStatistics()

            return prauc, rce, conf, max_pred, min_pred, avg

    
    # This method returns only the predictions
    #-------------------------------------------
    #           get_predictions(...)
    #-------------------------------------------
    # pool_tst:     Features of the test set
    #-------------------------------------------
    # As above, but without computing the scores
    #-------------------------------------------
    def get_prediction(self, pool_tst=None):
        Y_pred = None
        #Tries to load X and Y if not directly passed        
        if (pool_tst is None):
            print("No dataset provided.")
        if (self.model is None):
            print("No model trained yet.")
        else:
            #Preparing DMatrix
            #p_test = Pool(X_tst)

            #Making predictions            
            #Commented part gives probability but 2 columns
            # First column probability to be 0
            # Second column probability to be 1
            Y_pred = self.model.predict_proba(pool_tst)
            Y_pred = Y_pred[:,1]
            return Y_pred


    #This method loads a model
    #-------------------------
    # path: path to the model
    #-------------------------
    def load_model(self, path):      
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        print("Model correctly loaded.\n")
            
    
    # Returns/prints the importance of the features
    #-------------------------------------------------
    # verbose:   it also prints the features importance
    #-------------------------------------------------
    def get_feat_importance(self, verbose = False):
        
        #Getting feature importance
        importance = self.model.get_feature_importance(verbose=verbose)
        #for fstr_type parameter assign it something like = catboost.EFStrType.SharpValues
            
        return importance


    #-----------------------------------------------------
    #        Get the best iteration with ES
    #-----------------------------------------------------
    def getBestIter(self):
        return self.model.best_iteration_






















































#WHAT DOES THE CAT SAY?[semicit.]
#Miao
#Meaw
#Nyan
#Muwaa'
#Meo
#Meong
#Meu
#Miaou
#Miau
#Miauw
#Miaow
#Miyav
#Miav
#Mjau
#Miyau
#Mao
#Meogre
#Ngiiyaw