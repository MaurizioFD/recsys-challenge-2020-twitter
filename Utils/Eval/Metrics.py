import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, log_loss
from math import log
import xgboost as xgb
#---------------------------------------------------
# In this class are present the methods which define
# the evaluation metrics used in the challenge.
#---------------------------------------------------

class ComputeMetrics(object):
    def __init__(self, pred, gt):
        self.pred=pred.astype(np.float64)
        self.gt=gt.astype(np.float64)
    
    #--------------------------------------------------
    #   Code snippet provided for the challenge
    #--------------------------------------------------
    def compute_prauc(self):
        prec, recall, thresh = precision_recall_curve(self.gt, self.pred)
        prauc = auc(recall, prec)
        return prauc

    def calculate_ctr(self, gt):
        positive = len([x for x in gt if x == 1])
        ctr = positive/float(len(gt))
        return ctr

    def compute_rce(self):
        cross_entropy = log_loss(self.gt, self.pred)
        data_ctr = self.calculate_ctr(self.gt)
        strawman_cross_entropy = log_loss(self.gt, [data_ctr for _ in range(len(self.gt))])
        return (1.0 - cross_entropy/strawman_cross_entropy)*100.0
    #--------------------------------------------------


    #--------------------------------------------------
    #               ABOUT CONFUSION MATRIX
    #--------------------------------------------------
    # labels:        Labels to index the matrix.
    # sample_weight: Sample weights (array).
    # normalize:     Normalizes the confusion matrix
    #                 over the true (rows), predicted
    #                 conditions or all the population.
    #                 If None, no normalization
    #---------------------------------------------------
    def confMatrix(self,
                   labels=None, 
                   sample_weight=None, 
                   normalize=None):

        pred = self.binarize(self.pred)
        return confusion_matrix(self.gt,      
                                pred, 
                                labels=labels, 
                                sample_weight=sample_weight, 
                                normalize=normalize)

    #Makes a probability array binary
    def binarize(self, to_bin):
        threshold = 0.5
        to_bin=np.array(to_bin)
        #Why are symbols inverted, dunno but it works
        to_bin = np.where(to_bin < threshold, to_bin, 1)
        to_bin = np.where(to_bin > threshold, to_bin, 0)
        return to_bin
    #--------------------------------------------------

    #Computes some statistics about the prediction
    def computeStatistics(self):
        return max(self.pred), min(self.pred), np.mean(self.pred)


class CustomEvalXGBoost:

    def custom_eval(self, predt: np.ndarray, dtrain: xgb.DMatrix):
        eval_metric = float(log_loss(dtrain.get_label().astype(np.bool), predt.astype(np.float64)))
        return 'custom_log_loss', eval_metric


# Custom evaluation function for catboost
#----------------------------------------
# The entire object has to be passed to
# eval_metric parameter.
#----------------------------------------
class CustomEvalCATBoost(object):
    def is_max_optimal(self):
        # Returns whether great values of metric are better
        pass

    def evaluate(self, approxes, target, weight):
        # approxes is a list of indexed containers
        # (containers with only __len__ and __getitem__ defined),
        # one container per approx dimension.
        # Each container contains floats.
        # weight is a one dimensional indexed container.
        # target is a one dimensional indexed container.
        
        # weight parameter can be None.
        # Returns pair (error, weights sum)
        pass
    
    def get_final_error(self, error, weight):
        # Returns final value of metric based on error and weight
        pass