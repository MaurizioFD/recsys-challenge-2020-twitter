from abc import abstractmethod
from abc import ABC
import sys
from Utils.Base.RecommenderBase import RecommenderBase
import pickle
import base64
import datetime as dt


class RecommenderGBM(RecommenderBase):
    '''Defines the structure that all the models expose'''

    def __init__ (self, batch=False, name="recommenderbase", kind="NOT_GIVEN"):
        super(RecommenderGBM, self).__init__(
                batch=False,
                name=name,
                kind=kind)    

    '''
    #Maybe this works
    #------------------------------------------------------
    #               save_model(...)
    #------------------------------------------------------
    #filename:      [optional] defines name of saved model
    #path:          [optional] defines the path saved model
    #------------------------------------------------------
    def save_model(self, filename=None, path=None):
        #Defining the extension
        #Saving the model with premade name in working folder
        if (path is None) and (filename is None):
            date = str(dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
            if (self.batch is False):
                model_name = date+"_sround"+self.ext
                self.sround_model.save_model(model_name)
            else:
                model_name = date+"_batch"+self.ext
                self.batch_model.save_model(model_name)
            print("Model {0} saved successfully in working fodler.".format(model_name))

        #Saving model with given name in working folder
        elif (path is None) and (filename is not None):
            model_name = filename+self.ext
            if (self.batch is False):
                self.sround_model.save_model(model_name)
            else:
                self.batch_model.save_model(model_name)
            print("Model {0} saved successfully in working fodler.".format(model_name))

        #Saving model with given path but no name
        elif (path is not None) and (filename is None):
            date = str(dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
            if (self.batch is False):
                model_name = path+"/"+date+"_sround"+self.ext
                self.sround_model.save_model(model_name)
            else:
                model_name = path+"/"+date+"_sround"+self.ext
                self.batch_model.save_model(model_name)
            print("Model {0} saved successfully.".format(model_name))
        
        #Save with given path and filename
        else:
            model_name = path+"/"+filename+self.ext
            print(model_name)
            if (self.batch is False):
                self.sround_model.save_model(model_name)
            else:
                self.batch_model.save_model(model_name)
            print("Model {0} saved successfully.".format(model_name))
    '''


    #------------------------------------------------------
    #               save_model(...)
    #------------------------------------------------------
    #filename:      [optional] defines name of saved model
    #path:          [optional] defines the path saved model
    #------------------------------------------------------
    # TODO: Redoundant AF
    #------------------------------------------------------
    def save_model(self, filename=None, path=None):
        #Defining the extension
        #Saving the model with premade name in working folder
        if (path is None) and (filename is None):
            date = str(dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
            if (self.batch is False):
                model_name = date+"_sround"
                pickle.dump(self.model, open(model_name, "wb"))
            else:
                model_name = date+"_batch"
                pickle.dump(self.model, open(model_name, "wb"))
            print("Model {0} saved successfully in working fodler.".format(model_name))

        #Saving model with given name in working folder
        elif (path is None) and (filename is not None):
            model_name = filename
            if (self.batch is False):
                pickle.dump(self.model, open(model_name, "wb"))
            else:
                pickle.dump(self.model, open(model_name, "wb"))
            print("Model {0} saved successfully in working fodler.".format(model_name))

        #Saving model with given path but no name
        elif (path is not None) and (filename is None):
            date = str(dt.datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
            if (self.batch is False):
                model_name = path+"/"+date+"_sround"
                pickle.dump(self.model, open(model_name, "wb"))
            else:
                model_name = path+"/"+date+"_sround"
                pickle.dump(self.model, open(model_name, "wb"))
            print("Model {0} saved successfully.".format(model_name))
        
        #Save with given path and filename
        else:
            model_name = path+"/"+filename
            print(model_name)
            if (self.batch is False):
                pickle.dump(self.model, open(model_name, "wb"))
            else:
                pickle.dump(self.model, open(model_name, "wb"))
            print("Model {0} saved successfully.".format(model_name))
            
        return model_name




