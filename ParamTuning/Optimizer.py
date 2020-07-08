import numpy as np
import skopt
from skopt import gp_minimize
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import time
import datetime as dt
from ParamTuning.ModelInterface import ModelInterface


# ----------------------------------------------------
#           Simple skopt optimizer
# ----------------------------------------------------
# With this it is possible to run an optimization in 
# just a couple of commands.
# ----------------------------------------------------
# make_save:    saves reusable results
# make_log:     saves human readable results
# ----------------------------------------------------
# old batch (boolean) is now mode, an integer with:
# 0 --> monolithic optimization
# 1 --> batch optimization
# 2 --> nested cross validation
# ----------------------------------------------------
class Optimizer(object):
    def __init__(self, model_name,
                 kind,
                 mode=0,
                 auto_save=True,  # Saves the results at the end of optimization
                 make_save=True,  # Saves the results iteration per iteration
                 make_log=True,  # Make a human readable log iteration per iteration
                 path=None,  # Define path in which to save results of optimization
                 path_log=None):  # Path in which to save logs

        # Inputs
        self.model_name = model_name
        self.kind = kind
        self.mode = mode
        self.auto_save = auto_save  # saves the results without explictly calling the method
        self.make_save = make_save
        self.make_log = make_log
        self.path = path
        self.path_log = path_log
        # ModelInterface
        self.MI = None
        # Iteration counter
        self.iter_count = 0
        # Declaring result variable
        self.result = None

    # Setting the parameters of the optimizer
    def setParameters(self,
                      n_calls=20,
                      n_points=10000,
                      n_random_starts=10,
                      n_jobs=1,
                      # noise = 'gaussian',
                      noise=1e-7,
                      acq_func='gp_hedge',
                      acq_optimizer='auto',
                      random_state=None,
                      verbose=True,
                      n_restarts_optimizer=5,
                      xi=0.01,
                      kappa=1.96,
                      x0=None,
                      y0=None):

        self.n_point = n_points
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.n_jobs = n_jobs
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self.verbose = verbose
        self.xi = xi
        self.kappa = kappa
        self.noise = noise
        self.x0 = x0
        self.y0 = y0

        # Setting the model interface

    def defineMI(self, model_name=None, kind=None):
        if model_name is not None:
            self.model_name = model_name
        if kind is not None:
            self.kind = kind

        # Model interface
        self.MI = ModelInterface(model_name=self.model_name,
                                 kind=self.kind,
                                 mode=self.mode)

    # Defining the optimization method
    def optimize(self):

        # Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()

        # Setting filename
        if self.path is None:
            self.path = str(dt.datetime.now().strftime("%m_%d_%H_%M_%S"))

        # Checking if callback has to be called to make logs
        if (self.make_save is True):
            # Defining the callback function
            callback_function = self.callback_func
        else:
            callback_function = None

        # Making (or not) human readable logs
        self.MI.setSaveLog(self.make_log)

        # Path in which to save logs
        if self.path_log is not None:
            self.MI.setLogPath(self.path_log)

        self.result = gp_minimize(self.MI.getScoreFunc(),
                                  self.MI.getParams(),
                                  base_estimator=None,
                                  n_calls=self.n_calls,
                                  n_random_starts=self.n_random_starts,
                                  acq_func=self.acq_func,
                                  acq_optimizer=self.acq_optimizer,
                                  x0=self.x0,
                                  y0=self.y0,
                                  random_state=self.random_state,
                                  verbose=self.verbose,
                                  callback=callback_function,
                                  n_points=self.n_point,
                                  n_restarts_optimizer=self.n_restarts_optimizer,
                                  xi=self.xi,
                                  kappa=self.kappa,
                                  noise=self.noise,
                                  n_jobs=self.n_jobs)

        # Resetting the count for the logs
        self.MI.resetSaveLog()

        # Saving the obtained results
        if self.auto_save is True:
            self.saveRes(self.result)

        return self.result

    def callback_func(self, res):
        if self.make_save is True:
            self.saveRes(res)
        '''
        if self.make_log is True:
            self.saveLog(res)
        '''

    # Saving the results of the optimization with built-in method
    def saveRes(self, res):
        path = self.path + ".save.npz"
        # The only way to save this shit
        np.savez(path, x0=res.x_iters, y0=res.func_vals)
        # skopt.dump(res, path, store_objective=False)
        # print("Results {0} successfully saved.".format(path))

    '''
    #MOVED INTO MODEL INTERFACE CLASS
    #Method to save human readable logs
    def saveLog(self, res):
        #Parameters of the evaluation
        x = res.x_iters
        #Result of the evaluation
        y = res.func_vals
        #Taking the path provided
        path = self.path + ".log"
        #Get hyperparameter names
        p_names = self.MI.getParamNames()
        #Maybe check len(p_names) == len(x) here

        #Opening a file and writing into it the logs
        with open(path, 'a') as log:
            to_write = "ITERATION NUMBER " + str(self.iter_count) + "\n"
            log.write(to_write)
            for i in range(len(p_names)):
                to_write=str(str(p_names[i])+": "+str(x[self.iter_count][i])+"\n")
                log.write(to_write)

            #Written this way to be easily found
            to_write="--outcome--: "+str(y[self.iter_count])+"\n\n"
            log.write(to_write)

        #Increasing the iteration count
        self.iter_count = self.iter_count + 1
    '''

    # Loading model with built-in method (errors even with pickle)
    def loadModel(self, path=None):
        if (path is None):
            print("File path missing.")
        else:

            # The only way to save this shit
            model = np.load(path)

            # Splitting the model
            self.x0 = model['x0']
            self.y0 = model['y0']
            print(model['x0'])
            print(model['y0'])
            print("File {0} loaded successfully.".format(path))

    def loadModelHardCoded(self, path=None):
        # Splitting the model
        self.x0 = [
                265,
                46,
                0.15898209867759425,
                28,
                0.4267309590102383,
                0.7106015549429759,
                0.446213857304653,
                1.0,
                0.6629430145505582,
                0.6997710091846678,
                0.7287763868516478,
                22,
                2309
        ]

        self.y0 = [-5.47228466301597]

        print("Loaded Hard Coded Model as Prior Knowledge")

    # Load a custom dataset to train for the optimization
    def loadTrainData(self, X_train=None, Y_train=None, holder_train=None):
        # Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()

        self.MI.loadTrainData(X_train, Y_train, holder_train)

    # Load a custom dataset to test for the optimization
    def loadValData(self, X_val=None, Y_val=None, holder_val=None):
        # Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()

        self.MI.loadValData(X_val, Y_val, holder_val)

    # Load a custom dataset to test for the optimization
    def loadTestData(self, X_test=None, Y_test=None, holder_test=None):
        # Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()

        self.MI.loadTestData(X_test, Y_test, holder_test)

    # ---------------------------------------------------------------
    #                Batch train parameters
    # ---------------------------------------------------------------
    def batchTrain(self, tot_train_split, train_id):
        # Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()

        self.MI.batchTrain(tot_train_split, train_id)

    def batchVal(self, val_id):
        # Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()

        self.MI.batchVal(val_id)

    def batchTest(self, tot_test_split, test_id):
        # Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()

        self.MI.batchTest(tot_test_split, test_id)

    def setLabels(self, x_label, y_label, es_ncv=False):
        # Initializing model interface if it's None
        if self.MI is None:
            self.defineMI()

        self.MI.setLabels(x_label, y_label, es_ncv)

    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    #         Setting non tuned parameters for xgb
    # ---------------------------------------------------------------
    def setParamsXGB(self, verbosity=1,
                     process_type="default",
                     tree_method="auto",
                     objective="binary:logistic",
                     num_parallel_tree=4,
                     eval_metric="rmsle",
                     early_stopping_rounds=None):
        if self.MI is None:
            self.defineMI()

        self.MI.setParamsXGB(verbosity=verbosity,
                             process_type=process_type,
                             tree_method=tree_method,
                             objective=objective,
                             num_parallel_tree=num_parallel_tree,
                             eval_metric=eval_metric,
                             early_stopping_rounds=early_stopping_rounds)
    # --------------------------------------------------------------

    #---------------------------------------------------------------
    #         Setting non tuned parameters for lgb
    #---------------------------------------------------------------
    def setParamsLGB(self, verbosity=1, 
                        process_type="default", 
                        tree_method="auto", 
                        #Not in tuning dict
                        objective= 'binary',
                        num_threads= 4,
                        metric= ('cross_entropy','cross_entropy_lambda'),
                        num_parallel_tree=4, 
                        eval_metric="rmsle", 
                        early_stopping_rounds=None,
                        is_unbalance=False):
        if self.MI is None:
            self.defineMI()
        
        self.MI.setParamsLGB(verbosity=verbosity,
                             process_type=process_type,
                             tree_method=tree_method,
                             objective=objective,
                             num_parallel_tree=num_parallel_tree,
                             eval_metric=eval_metric,
                             early_stopping_rounds=early_stopping_rounds,
                             is_unbalance=is_unbalance)
    #---------------------------------------------------------------

    #---------------------------------------------------------------
    #         Setting non tuned parameters for cat
    #---------------------------------------------------------------
    def setParamsCAT(self, verbosity= 1,
                           boosting_type= "Plain",
                           model_shrink_mode= "Constant",
                           leaf_estimation_method= "Newton",
                           bootstrap_type= "Bernoulli",
                           early_stopping_rounds= 5):
        if self.MI is None:
            self.defineMI()

        self.MI.setParamsCAT(verbosity= verbosity,
                             boosting_type= boosting_type,
                             model_shrink_mode= model_shrink_mode,
                             leaf_estimation_method= leaf_estimation_method,
                             bootstrap_type= bootstrap_type,
                             early_stopping_rounds= early_stopping_rounds)
    #---------------------------------------------------------------


    def setCategoricalFeatures(self,categorical_features=None):
        if self.MI is None:
            self.defineMI()
        self.MI.setCategoricalFeatures(categorical_features)
