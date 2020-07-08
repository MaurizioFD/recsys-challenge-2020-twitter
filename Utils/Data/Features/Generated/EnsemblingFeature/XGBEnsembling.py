import pathlib

import Models.GBM.XGBoost as wrapper
import Utils.Data as data
from Utils.Data.Features.Generated.EnsemblingFeature.EnsemblingFeatureAbstract import EnsemblingFeatureAbstract
import xgboost as xgb
import random
import pandas as pd


class XGBEnsembling(EnsemblingFeatureAbstract):
    path = "xgb_ensembling"

    def __init__(self,
                 dataset_id: str,
                 df_train: pd.DataFrame,
                 df_train_label: pd.DataFrame,
                 df_to_predict: pd.DataFrame,
                 param_dict: dict
                 ):
        self.dataset_id = dataset_id
        self.feature_name = f"xgb_ensembling"
        super().__init__(df_train, df_train_label, df_to_predict, param_dict)

    def _get_dataset_id(self):
        return self.dataset_id

    def _get_path(self):
        return self.path

    def _get_feature_name(self):
        return self.feature_name

    def _load_model(self):
        model = wrapper.XGBoost()
        model.load_model(self.model_path)
        return model

    def _train_and_save(self):
        # Generate a random number
        random_n = random.random()
        # Initiate XGBoost wrapper
        xgb_wrapper = wrapper.XGBoost(
            tree_method="gpu_hist",
            num_rounds=self.param_dict['num_rounds'],
            max_depth=self.param_dict['max_depth'],
            min_child_weight=self.param_dict['min_child_weight'],
            colsample_bytree=self.param_dict['colsample_bytree'],
            learning_rate=self.param_dict['learning_rate'],
            reg_alpha=self.param_dict['reg_alpha'],
            reg_lambda=self.param_dict['reg_lambda'],
            scale_pos_weight=self.param_dict['scale_pos_weight'],
            gamma=self.param_dict['gamma'],
            subsample=self.param_dict['subsample'],
            base_score=self.param_dict['base_score'],
            max_delta_step=self.param_dict['max_delta_step'],
            num_parallel_tree=self.param_dict['num_parallel_tree']
        )
        # Cache the train matrix as libsvm
        data.DataUtils.cache_dataset_as_svm(f"temp_ensembling_{random_n}", self.df_train, self.df_train_label)
        # Load the train matrix + external memory
        # train = xgb.DMatrix(f"temp_ensembling_{random_n}.svm#temp_ensembling_{random_n}.cache")
        train = xgb.DMatrix(f"temp_ensembling_{random_n}.svm")
        # Overwrite the feature names for consistency
        train.feature_names = self.df_train.columns
        # Fit the model
        xgb_wrapper.fit(dmat_train=train)
        # Create the directory (where the model is saved) if it does not exist
        pathlib.Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        # Save the model
        xgb_wrapper.save_model(filename = self.model_path)

    def create_feature(self):
        # Load the model
        model = self._get_model()
        # Generate a random number
        random_n = random.random()
        # Cache the train matrix as libsvm
        data.DataUtils.cache_dataset_as_svm(f"temp_ensembling_test_{random_n}", self.df_to_predict[model.model.feature_names], no_fuck_my_self=True)
        # Load the train matrix + external memory
        test = xgb.DMatrix(f"temp_ensembling_test_{random_n}.svm")
        # Overwrite the feature names for consistency
        test.feature_names = model.model.feature_names
        # Predict the labels
        predictions = model.get_prediction(test)
        # Encapsulate the labels
        result = pd.DataFrame(predictions, index=self.df_to_predict.index)
        # Sort the labels
        result.sort_index(inplace=True)
        # Save the result
        self.save_feature(result)
