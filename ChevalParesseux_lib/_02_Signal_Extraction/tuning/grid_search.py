from ..tuning import common as com
from ...utils import metrics as met
from ...utils import calculations as calc

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Union, List, Self



#! ==================================================================================== #
#! ================================= Main Function ==================================== #
class Classifier_gridSearch(com.PredictorTuning):
    def __init__(
        self, 
        n_jobs: int = 1,
        random_state: int = 72,
    ) -> None:
        
        super().__init__(n_jobs=n_jobs)
        
        self.random_state = random_state
        np.random.seed(random_state)
        
        # ======= II. Results =======
        self.best_params = None
        self.best_score = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        random_search: bool = False,
        n_samples: int = 10,
    ) -> Self:
        self.params = {
            'random_search': random_search,
            'n_samples': n_samples,
        }
        
        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self,
        data: Union[pd.DataFrame, list],
    ) -> Union[pd.DataFrame, list]:
        
        return data
    
    #?____________________________________________________________________________________ #
    def train_model(
        self, 
        model: object,
        model_params: dict,
        features_matrix: pd.DataFrame,
        target_vector: pd.Series,
    ) -> object:
        # ======= I. Initiate the model =======
        trained_model = model(n_jobs=self.n_jobs).set_params(**model_params)
        
        # ======= II. Train the model =======
        trained_model.fit(features_matrix, target_vector)
        
        return trained_model
    
    #?____________________________________________________________________________________ #
    def get_params_results(
        self, 
        model: object,
        criteria: str, 
        features_matrix: pd.DataFrame,
        target_vector: pd.Series,
    ) -> float:
        # ======= I. Make predictions =======
        predictions = model.predict(features_matrix)
        predictions = pd.Series(predictions)
        
        # ======= II. Calculate the metrics =======
        classes_labels = pd.Series(target_vector).unique()
        metrics = met.generate_classification_metrics(predictions=predictions, labels=target_vector, classes=classes_labels)
        measure = metrics[criteria]

        return measure
    
    #?____________________________________________________________________________________ #
    def fit(
        self,
        model: object,
        grid_universe: dict,
        criteria: str,
        data: List[tuple],
    ) -> Self:
        # ======= I. Extract the parameters =======
        if self.params['random_search']:
            grid_universe = calc.get_random_dict_universe(grid_universe, self.params['n_samples'], self.random_state)
        else:
            grid_universe = calc.get_dict_universe(grid_universe)
        
        # ======= II. Process the data =======
        
        
    #?____________________________________________________________________________________ #
    def fit(self):
        # ======= 0. Extract the parameters =======
        params = self.gridSearch_params
        criteria = params["criteria"]
        n_folds = params["n_folds"]
        balanced_training = params["balanced_training"]
        
        # ======= I. Create the folds ======= 
        if self.folds is None or self.balanced_folds is None:
            folds, balanced_folds = self.transform_data(n_folds=n_folds)
        
            if not balanced_training:
                balanced_folds = folds.copy()
            
            self.folds = folds
            self.balanced_folds = balanced_folds
        
        else:
            folds = self.folds
            balanced_folds = self.balanced_folds
        
        # ======= II. Calibrate Models =======
        params_list = com.extract_universe(self.gridUniverse)
        results = Parallel(n_jobs=self.n_jobs)(delayed(self.test_params)(params=params, criteria=criteria, folds=folds, balanced_folds=balanced_folds) for params in params_list)
        
        best_results = max(results, key=lambda x: x['score'])
        best_params = best_results['params']
        
        # ======= IV. Save the results =======
        best_params = best_results['params']
        best_score = best_results['score']

        self.best_params = best_params
        self.best_score = best_score
        
        return best_params, best_score
        

# Classifier_bayesian
# Classifier_genetic