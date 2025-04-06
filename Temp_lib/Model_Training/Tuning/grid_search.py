from ..Tuning import common as com
from ..Tuning import set_generator as setg

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
class GridSearchCV(com.GridSearch):
    def __init__(self, model, training_data: pd.DataFrame):
        super().__init__(model)

        self.training_data = training_data
        self.set_generator = None

        self.params_grid = None
        self.params_sets = None
        self.non_feature_columns = None

        self.folds = None
        self.balanced_folds = None
    
    #?____________________________________________________________________________________ #
    def transform_data(self, n_folds: int):
        folds = self.set_generator.create_folds(df=self.training_data, n_folds=n_folds)
        balanced_folds = self.set_generator.balance_data(df_list=folds)

        self.folds = folds
        self.balanced_folds = balanced_folds

        return folds, balanced_folds

    #?____________________________________________________________________________________ #
    def set_params_grid(self, params_grid: dict, non_feature_columns: list):
        self.params_grid = params_grid
        self.non_feature_columns = non_feature_columns

    #?____________________________________________________________________________________ #
    def set_params_sets(self, params_sets: dict):
        self.params_sets = params_sets
        self.set_generator = setg.SetGenerator(self.training_data)
        self.set_generator.set_params(**self.params_sets)
        
    #?____________________________________________________________________________________ #
    def fit(self, n_folds: int):

        # ======= I. Create the folds & extract Parameters ======= 
        folds, balanced_folds = self.transform_data(n_folds=n_folds)
        params_list = com.extract_universe(self.params_grid)

        # ======= II. Train the model on each fold =======
        for fold_idx in range(n_folds):
            # II.1 Extract the training and testing folds
            testing_fold = folds[fold_idx]
            training_fold = balanced_folds[:fold_idx] + balanced_folds[fold_idx + 1:]
            training_fold = pd.concat(training_fold, ignore_index=True, axis=0).reset_index(drop=True)

            # II.2 Extract the features and target variable
            X_train = training_fold.drop(columns=self.non_feature_columns)
            y_train = training_fold[self.set_generator.labels_name]

            X_test = testing_fold.drop(columns=self.non_feature_columns)
            y_test = testing_fold[self.set_generator.labels_name]
            
            # II.3 Train the model on each parameter set
            for params in params_list:
                self.model.set_params(**params)
                self.model.fit(X_train, y_train)

                # II.3.3 Score the model
                
        





    
