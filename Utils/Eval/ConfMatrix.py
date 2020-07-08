from sklearn.metrics import confusion_matrix
import numpy as np
#--------------------------------------------------
#               ABOUT CONFUSION MATRIX
#--------------------------------------------------
# gt:            Ground truth of predictions.
# pred:          Predicted values.
# labels:        Labels to index the matrix.
# sample_weight: Sample weights (array).
# normalize:     Normalizes the confusion matrix
#                 over the true (rows), predicted
#                 conditions or all the population.
#                 If None, no normalization
#---------------------------------------------------

#Returns the confusion matrix
def confMatrix(gt, 
                pred, 
                labels=None, 
                sample_weight=None, 
                normalize=None):

    
    pred = binarize(pred)
    

    return confusion_matrix(gt,      
                            pred, 
                            labels=labels, 
                            sample_weight=sample_weight, 
                            normalize=normalize)


#Makes a probability array binary
def binarize(to_bin):
    threshold = 0.5
    to_bin=np.array(to_bin)
    #Why are symbols inverted, dunno but it works
    to_bin = np.where(to_bin < threshold, to_bin, 1)
    to_bin = np.where(to_bin > threshold, to_bin, 0)
    return to_bin
