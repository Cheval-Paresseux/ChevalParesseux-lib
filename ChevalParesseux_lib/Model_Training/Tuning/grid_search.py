from ..Tuning import common as com
from ..Tuning import set_generator as setg
from ...utils import classificationMetrics as clsmetrics 

import pandas as pd
import numpy as np
from joblib import Parallel, delayed



class GridSearchCV(com.GridSearch):
    def __init__(
        self, 
        model, 
        training_data: pd.DataFrame, 
        n_jobs: int = 1
    ):
        # ======= I. Input Parameters =======
        self.model = model
        self.training_data = training_data
        self.n_jobs = n_jobs
        
        # ======= II. Data Processing Parameters =======
        self.set_generator = None
        self.params_grid = None
        self.params_sets = None
        self.non_feature_columns = None
        
        # ======= III. Processed data =======
        self.folds = None
        self.balanced_folds = None
        
        # ======= IV. Results =======
        self.best_params = None
        self.best_score = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        params_grid: dict, 
        params_sets: dict, 
        non_feature_columns: list
    ):
        # ======= I. Set the parameters =======
        self.params_grid = params_grid
        self.params_sets = params_sets
        self.non_feature_columns = non_feature_columns
        
        # ======= II. Initialize the set generator =======
        self.set_generator = setg.SetGenerator(training_df=self.training_data, n_jobs=self.n_jobs, random_state=72).set_params(**self.params_sets)
    
    #?____________________________________________________________________________________ #
    def transform_data(
        self, 
        n_folds: int
    ):
        # ======= I. Create the folds for cross-testing =======
        folds = self.set_generator.create_folds(df=self.training_data, n_folds=n_folds)
        
        # ======= II. Create the balanced folds for training =======
        balanced_folds = self.set_generator.balance_data(df_list=folds)

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
            y_train = training_fold[self.set_generator.params['labels_name']]

            X_test = testing_fold.drop(columns=self.non_feature_columns)
            y_test = testing_fold[self.set_generator.params['labels_name']]

            model.fit(X_train, y_train)

            # II.3.3 Score the model
            predictions = pd.Series(model.predict(X_test))
            metrics = clsmetrics.get_classification_metrics(predictions=predictions, labels=y_test, classes=[-1, 1])
            measure = metrics[criteria]
            measures.append(measure)
        
        # ======= III. Calculate the mean score =======
        cv_score = np.mean(measures)
        results = {'params': params, 'score': cv_score}
        
        return results
        
    #?____________________________________________________________________________________ #
    def fit(
        self,
        criteria: str,
        n_folds: int,
        balanced_training: bool = True
    ):
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
        
        # ======= II. Test the model for each Parameters set =======
        params_list = com.extract_universe(self.params_grid)
        results = Parallel(n_jobs=self.n_jobs)(delayed(self.test_params)(params=params, criteria=criteria, folds=folds, balanced_folds=balanced_folds) for params in params_list)
        
        # ======= III. Sort the results =======
        best_results = max(results, key=lambda x: x['score'])
        best_params = best_results['params']
        
        # ======= IV. Save the results =======
        self.results = results
        self.best_params = best_params
        self.best_score = best_results['score']
        
        return best_params
        
        
        
        
        
        

                
        





    
