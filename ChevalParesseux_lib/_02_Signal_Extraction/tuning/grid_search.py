from ..tuning import common as com
from ... import utils

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Union, List, Self
from tqdm import tqdm



#! ==================================================================================== #
#! ================================= Main Function ==================================== #
class Classifier_gridSearch(com.PredictorTuning):
    """
    Grid search for tuning classifiers.
    
    This class implements a grid search algorithm for tuning classifiers.
    It allows for both random and exhaustive search of hyperparameters.
    """
    #?_____________________________ Initialization methods _______________________________ #
    def __init__(
        self, 
        n_jobs: int = 1,
        random_state: int = 72,
    ) -> None:
        """
        Constructor for the Classifier_gridSearch class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to use during computation.
            - random_state (int): Random state for reproducibility.
        """
        # ======= I. Initialize Class =======
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
        """
        Sets the parameters for the grid search.
        
        Parameters:
            - random_search (bool): If True, performs a random search instead of an exhaustive search.
            - n_samples (int): Number of samples to draw for random search.
        
        Returns:
            - Self: The instance of the class with the parameters set.
        """
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
        """
        Processes the input data and returns a DataFrame or list.
        
        Parameters:
            - data (pd.DataFrame | list): The input data to be processed.
        
        Returns:
            - Union[pd.DataFrame, list]: A DataFrame or list containing the processed data.
        """
        return data
    
    #?______________________________ Building methods ____________________________________ #
    def fit_model(
        self, 
        model: object,
        model_params: dict,
        features_matrix: pd.DataFrame,
        target_vector: pd.Series,
    ) -> object:
        """
        Fits the model with the given parameters.
        
        Parameters:
            - model (object): The model to be fitted.
            - model_params (dict): The parameters for the model.
            - features_matrix (pd.DataFrame): The features matrix for training.
            - target_vector (pd.Series): The target vector for training.
        
        Returns:
            - object: The fitted model.
        """
        # ======= I. Initiate the model =======
        fitted_model = model(n_jobs=self.n_jobs, random_state=self.random_state).set_params(**model_params)
        
        # ======= II. Train the model =======
        fitted_model = fitted_model.fit(features_matrix, target_vector)
        
        return fitted_model
    
    #?____________________________________________________________________________________ #
    def get_score(
        self, 
        model: object,
        criteria: str, 
        features_matrix: pd.DataFrame,
        target_vector: pd.Series,
    ) -> float:
        """
        Calculate the score of the model using the given criteria.
        
        Parameters:
            - model (object): The fitted model.
            - criteria (str): The criteria for scoring the model.
            - features_matrix (pd.DataFrame): The features matrix for testing.
            - target_vector (pd.Series): The target vector for testing.
        
        Returns:
            - float: The score of the model.
        """
        # ======= I. Make predictions =======
        predictions = model.predict(features_matrix)
        predictions = pd.Series(predictions, index=features_matrix.index)
        
        # ======= II. Calculate the metrics =======
        classes_labels = pd.Series(target_vector).unique()
        metrics = utils.generate_classification_metrics(predictions=predictions, labels=target_vector, classes=classes_labels)
        measure = metrics[criteria]

        return measure
    
    #?____________________________________________________________________________________ #
    def evaluate_params(
        self, 
        model: object, 
        model_params: dict, 
        criteria: str,
        data: List[tuple], 
    ) -> dict:
        """
        Evaluate the model with the given parameters and data.
        
        Parameters:
            - model (object): The model to be evaluated.
            - model_params (dict): The parameters for the model.
            - criteria (str): The criteria for scoring the model.
            - data (list): The data to be used for evaluation.
        
        Returns:
            - dict: A dictionary containing the parameters and score.
        """
        # ======= Case 1 : no folds defined =======
        if len(data) == 1:
            features_matrix, target_vector = data[0]
            fitted_model = self.fit_model(model, model_params, features_matrix, target_vector)
            score = self.get_score(fitted_model, criteria, features_matrix, target_vector)
        
        # ======= Case 2 : with folds defined =======
        else:
            score = 0
            weights = 0
            for idx in range(len(data)):
                train_fold = data[idx]
                test_folds = data[:idx] + data[idx+1:]

                features_matrix = train_fold[0]
                target_vector = train_fold[1]
                fitted_model = self.fit_model(model, model_params, features_matrix, target_vector)

                for test_fold in test_folds:
                    test_features_matrix = test_fold[0]
                    test_target_vector = test_fold[1]
                    
                    weight = len(test_features_matrix)
                    fold_score = self.get_score(fitted_model, criteria, test_features_matrix, test_target_vector)

                    score += weight * fold_score
                    weights += weight
            
            score /= weights
        
        return {'params': model_params, 'score': score}

    #?__________________________________ User methods ____________________________________ #
    def fit(
        self,
        model: object,
        grid_universe: dict,
        criteria: str,
        data: List[tuple],
    ) -> Self:
        """
        Fit the grid search model.
        
        Parameters:
            - model (object): The model to be tuned.
            - grid_universe (dict): The grid universe of parameters.
            - criteria (str): The criteria for scoring the model.
            - data (list): The data to be used for evaluation.
        
        Returns:
            - Self: The instance of the class with the best parameters and score.
        """
        # ======= I. Extract the parameters =======
        if self.params['random_search']:
            grid_universe = utils.get_random_dict_universe(grid_universe, self.params['n_samples'], self.random_state)
        else:
            grid_universe = utils.get_dict_universe(grid_universe)
        
        # ======= II. Run the grid search =======
        grid_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.evaluate_params)(model, params, criteria, data)
            for params in tqdm(grid_universe)
        )
        
        # ======= III. Save the results =======
        best_results = max(grid_results, key=lambda x: x['score'])
        best_params = best_results['params']
        best_score = best_results['score']

        self.best_params = best_params
        self.best_score = best_score

        return self
    
    #?____________________________________________________________________________________ #
    def extract(
        self,
        model: object,
        data: tuple,
    ) -> object:
        """
        Extract the model with the best parameters.
        
        Parameters:
            - model (object): The model to be extracted.
            - data (tuple): The data to be used for extraction.
        
        Returns:
            - object: The fitted model.
        """
        # ======= I. Initialize the model =======
        fitted_model = model(n_jobs=self.n_jobs).set_params(**self.best_params)

        # ======= II. Train the model =======
        features_matrix = data[0]
        target_vector = data[1]
        fitted_model.fit(features_matrix, target_vector)
        
        return fitted_model
                
        
# class Classifier_bayesianSearch(com.PredictorTuning):
# class Classifier_geneticSearch(com.PredictorTuning):