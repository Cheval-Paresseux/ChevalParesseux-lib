from ..Tuning import common as com
from ...utils import metrics as clsmetrics 

import pandas as pd
import numpy as np
from joblib import Parallel, delayed


#! ==================================================================================== #
#! ================================= Main Function ==================================== #
class linearCV_GridSearch(com.GridSearch):
    """
    Cross Validation Grid Search with temporal linear folds.
    
    This class implements a grid search cross-validation approach to tune the hyperparameters of a given model. 
    It inherits from the GridSearch class and provides methods to : 
        - Set parameters for the grid search.
        - Transform the data into folds for cross-validation.
        - Test the model with different parameter combinations.
        - Fit the model and find the best parameters based on a specified scoring criteria.
    """
    def __init__(
        self, 
        model: object,
        training_data: pd.DataFrame, 
        n_jobs: int = 1
    ):
        """
        Initializes the GridSearchCV class with the model and training data.
        
        Parameters:
            - model: The machine learning model to be tuned.
            - training_data (pd.DataFrame): The data to be used for training and testing the model.
            - n_jobs (int): The number of jobs to run in parallel. Default is 1.
        """
        # ======= I. Input Parameters =======
        self.model = model
        self.training_data = training_data
        self.n_jobs = n_jobs
        
        # ======= II. Data Processing Parameters =======
        self.splitSample_model = None
        self.splitSample_params = None
        
        # ======= III. Grid Parameters =======
        self.gridSearch_params = None #{"criteria": "f1_score", "n_folds": 5, "balanced_training": False}
        self.gridUniverse = None

        self.non_feature_columns = None

        # ======= IV. Processed data =======
        self.folds = None
        self.balanced_folds = None
        
        # ======= IV. Results =======
        self.best_params = None
        self.best_score = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        splitSample_model: object,
        splitSample_params: dict, 
        gridSearch_params: dict, 
        gridUniverse: dict,
        non_feature_columns: list
    ):
        """
        Defines the parameters for the grid search and folds generation.
        
        Parameters:
            - splitSample_model: The model used for splitting the data into folds.
            - splitSample_params (dict): The parameters for the splitSample model.
            - gridSearch_params (dict): The parameters for the grid search.
            - gridUniverse (dict): The grid universe for the parameter combinations.
            - non_feature_columns (list): The columns that are not features in the training data.
        """
        # ======= I. Split And Sample Parameters =======
        self.splitSample_model = splitSample_model
        self.splitSample_params = splitSample_params
        
        # ======= II. Grid Search Parameters =======
        self.gridSearch_params = gridSearch_params
        
        # ====== III. Grid Universe =======
        self.gridUniverse = gridUniverse
        
        # ======= IV. Annexe =======
        self.non_feature_columns = non_feature_columns
        
        return self
    
    #?____________________________________________________________________________________ #
    def transform_data(
        self, 
        n_folds: int
    ):
        """
        Generates the folds for cross-validation.
        
        Parameters:
            - n_folds (int): The number of folds to be created.
        
        Returns:
            - folds (list): The generated folds raw from the training data.
            - balanced_folds (list): The balanced folds, resampled from the training data.
        """
        # ======= I. Create the folds for cross-testing =======
        splitSample_generator = self.splitSample_model(training_df=self.training_data, n_jobs=self.n_jobs, random_state=72).set_params(**self.splitSample_params)
        folds, balanced_folds = splitSample_generator.extract(df=self.training_data, n_folds=n_folds)

        # ======= III. Save the folds =======
        self.folds = folds
        self.balanced_folds = balanced_folds

        return folds, balanced_folds
    
    #?____________________________________________________________________________________ #
    def test_params(
        self, 
        params: dict, 
        criteria: str, 
        folds: list, 
        balanced_folds: list
    ):
        """
        Tests the model with a specific set of parameters and evaluates its performance.
        
        Parameters:
            - params (dict): The parameters to be tested.
            - criteria (str): The scoring criteria to be used for evaluation.
            - folds (list): The generated folds raw from the training data.
            - balanced_folds (list): The balanced folds, resampled from the training data.
        
        Returns:
            - results (dict): A dictionary containing the tested parameters and the corresponding score.
        """
        # ======= I. Initiate the model =======
        model = self.model(n_jobs=self.n_jobs)
        model.set_params(**params)
        
        # ======= II. Test the model on different folds =======
        measures = []
        
        for fold_idx in range(len(folds)):
            # II.1 Extract the training and testing folds
            testing_fold = folds[fold_idx]
            training_fold = balanced_folds[:fold_idx] + balanced_folds[fold_idx + 1:]
            training_fold = pd.concat(training_fold, ignore_index=True, axis=0).reset_index(drop=True)

            # II.2 Extract the features and target variable
            X_train = training_fold.drop(columns=self.non_feature_columns)
            y_train = training_fold["label"]

            X_test = testing_fold.drop(columns=self.non_feature_columns)
            y_test = testing_fold["label"]

            model.fit(X_train, y_train)

            # II.3 Score the model
            predictions = pd.Series(model.predict(X_test))
            
            classes_labels = pd.Series(y_train).unique()
            metrics = clsmetrics.get_classification_metrics(predictions=predictions, labels=y_test, classes=classes_labels)
            measure = metrics[criteria]
            measures.append(measure)
            
        # ======= IV. Calculate the mean score =======
        cv_score = np.mean(measures)
        results = {'params': params, 'score': cv_score}

        return results
        
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
        

