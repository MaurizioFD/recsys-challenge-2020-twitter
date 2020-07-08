from abc import abstractmethod
from abc import ABC
import sys



class RecommenderBase(ABC):
    '''Defines the structure that all the models expose'''

    def __init__ (self, batch=False, name="recommenderbase", kind="NOT_GIVEN"):

        super(RecommenderBase, self).__init__()

        self.name = name
        self.batch = batch
        self.kind = kind
        
    @abstractmethod
    def fit(self):
        '''
        Fit the model on the data (train). 
        Inherited class should extend this method in appropriate way.
        '''
        raise NotImplementedError("error :)")

    @abstractmethod
    def evaluate(self):
        '''
        Compute the predictions then performs the evaluation of the predicted values. 
        Inherited class should extend this method in appropriate way.
        '''
        raise NotImplementedError("errror :))")

    @abstractmethod
    def get_prediction(self):
        '''
        Compute the predictions without performing the evaluation. 
        Inherited class should extend this method in appropriate way.
        '''
        raise NotImplementedError("error :)))")

    @abstractmethod
    def load_model(self):
        '''
        Load a compatible model. 
        Inherited class should extend this method in appropriate way.
        '''
        raise NotImplementedError("error :)))")

   
    #--------------------------------------------------------------------
    # Still to be implemented, need dictionaries
    #--------------------------------------------------------------------
    # The method to save predictions in a submittable form in a file can 
    # be implemented here
    def save_in_file(self):
        # Retrieve the path to the ditionaries and load them
        # Rebuild the form <usr_id><itm_id><score>
        # Save this shit into a file
        pass
    #--------------------------------------------------------------------


        

