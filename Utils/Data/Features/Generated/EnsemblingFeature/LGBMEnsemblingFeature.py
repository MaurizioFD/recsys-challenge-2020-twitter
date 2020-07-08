import pathlib
import Models.GBM as wrapper
from Utils.Data.Features.Generated.EnsemblingFeature.EnsemblingFeatureAbstract import EnsemblingFeatureAbstract
import pandas as pd
import time
import hashlib

class LGBMEnsemblingFeature(EnsemblingFeatureAbstract):
    path = "lgbm_ensembling"

    def __init__(self,
                 dataset_id: str,
                 df_train: pd.DataFrame,
                 df_train_label: pd.DataFrame,
                 df_to_predict: pd.DataFrame,
                 param_dict: dict,
                 categorical_features_set: set,
                 ):

        features = list(df_train.columns)
        label = list(df_train_label.columns)
        hash_features = hashlib.md5(repr(features).encode('utf-8')).hexdigest()
        hash_label = hashlib.md5(repr(label).encode('utf-8')).hexdigest()
        hash_param_dict = hashlib.md5(repr(param_dict.items()).encode('utf-8')).hexdigest()
        hashcode = f"{hash_features}_{hash_label}_{hash_param_dict}"
        self.feature_name = f"lgbm_blending_{hashcode}"
        self.dataset_id = dataset_id
        self.categorical_features_set = categorical_features_set
        super().__init__(df_train, df_train_label, df_to_predict, param_dict)

    def _get_dataset_id(self):
        return self.dataset_id

    def _get_path(self):
        return self.feature_name

    def _get_feature_name(self):
        return self.feature_name

    def _load_model(self):
        model = wrapper.LightGBM.LightGBM()
        model.load_model(self.model_path)
        return model

    def _train_and_save(self):
        # LGBM Training
        training_start_time = time.time()

        lgbm_wrapper = wrapper.LightGBM.LightGBM(**self.param_dict)

        lgbm_wrapper.fit(X=self.df_train, Y=self.df_train_label, categorical_feature=self.categorical_features_set)
        print(f"Training time: {time.time() - training_start_time} seconds")

        # Create the directory (where the model is saved) if it does not exist
        pathlib.Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        # Save the model
        lgbm_wrapper.save_model(filename=self.model_path)

    def create_feature(self):
        # Load the model
        model = self._get_model()
        # Predict the labels
        predictions = model.get_prediction(self.df_to_predict)
        # Encapsulate the labels
        result = pd.DataFrame(predictions, index=self.df_to_predict.index)
        # Sort the labels
        result.sort_index(inplace=True)
        # Save the result
        self.save_feature(result)
