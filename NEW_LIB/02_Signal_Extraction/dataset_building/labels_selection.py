from ..dataset_building import common as com

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Union, Self

#! ==================================================================================== #
#! ================================= Main Function ==================================== #
class TemporalUniquenessSplitter(com.DatasetBuilder):
    def __init__(
        self, 
        n_jobs: int = 1, 
        random_state: int = 72
    ):
        super().__init__(n_jobs=n_jobs)
        self.random_state = random_state
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        label_column: list = ['label'],
        price_column: list = ['close'],
        n_samples: list = [1000],
        replacement: list = [False],
        vol_window: list = [10],
        upper_barrier: list = [0.5],
        vertical_barrier: list = [20],
    ) -> Self:
        
        self.params = {
            'label_column': label_column,
            'price_column': price_column,
            'n_samples': n_samples,
            'replacement': replacement,
            'vol_window': vol_window,
            'upper_barrier': upper_barrier,
            'vertical_barrier': vertical_barrier,
        }
        
        return self
        
    #?____________________________________________________________________________________ #
    def balance_data(
        self, 
        df_list: list
    ) -> pd.DataFrame:
        
        np.random.seed(self.random_state)

        balanced_dfs = []
        for df in df_list:
            # ======= I. Extract the sample weights =======
            training_df = df.copy()
            training_df.dropna(inplace=True)
            
            sample_weights, _ = get_samples_weights(
                labels_series=training_df[self.params['labels_name']], 
                price_series=training_df[self.params['price_name']], 
                vol_window=self.params['vol_window'], 
                upper_barrier=self.params['upper_barrier'], 
                vertical_barrier=self.params['vertical_barrier'],
                labels_name=self.params['labels_name'],
                price_name=self.params['price_name'],
                n_jobs=self.n_jobs
            )
            training_df = pd.concat([training_df, sample_weights], axis=1)
            training_df = training_df.dropna(axis=0)

            # ======= II. Separate Classes =======
            available_labels = training_df[self.params['labels_name']].unique()
            labels_specific_dfs = [training_df[training_df[self.params['labels_name']] == label].copy() for label in available_labels]

            # ======= III. Sample Each Class =======
            nb_labels = len(labels_specific_dfs)
            n_samples = self.params['n_samples'] // nb_labels
            
            sampled_dfs = []
            for unique_df in labels_specific_dfs:
                unique_df['sample_weights'] = unique_df['sample_weights'] / unique_df['sample_weights'].sum()
                
                if not self.params['replacement']:
                    n_samples = min(n_samples, unique_df.shape[0])
                    
                sampled_indices = np.random.choice(unique_df.index, size=n_samples, replace=self.params['replacement'], p=unique_df["sample_weights"])
                
                df_sampled = unique_df.loc[sampled_indices].reset_index(drop=True)
                sampled_dfs.append(df_sampled)
            
            # ======= IV. Concatenate the Sampled DataFrames =======
            df_sampled = pd.concat(sampled_dfs, axis=0).reset_index(drop=True)
            df_sampled = df_sampled.drop(columns=['sample_weights'])
            balanced_dfs.append(df_sampled)
        
        # ======= IV. Store the Balanced data =======
        self.balanced_dfs = balanced_dfs
        
        return balanced_dfs

    #?____________________________________________________________________________________ #
    def create_folds(
        self, 
        df: pd.DataFrame, 
        n_folds: int = 5
    ) -> list:
        """
        Creates folds for cross-validation.
        
        Parameters:
            - df (pd.DataFrame): The input DataFrame to be split into folds.
            - n_splits (int): The number of folds to create.
        
        Returns:
            - folds (list): A list of DataFrames representing the folds.
        """
        # ======= I. Get the split indexes =======
        size_df = df.shape[0]
        fold_size = (size_df - n_folds * self.params['vertical_barrier']) // n_folds  # Adjusted to avoid leakage
        end_indexes = [(i + 1) * (fold_size + self.params['vertical_barrier']) - self.params['vertical_barrier'] for i in range(n_folds)]
        start_indexes = [0] + [e + self.params['vertical_barrier'] for e in end_indexes[:-1]]
        
        # ====== II. Create the folds =======
        folds = []
        for start, end in zip(start_indexes, end_indexes):
            fold = df.iloc[start:end].copy()
            fold.reset_index(drop=True, inplace=True)
            folds.append(fold)
        
        # ====== III. Store the folds =======
        self.folds = folds
        
        return folds
    
    #?____________________________________________________________________________________ #
    def extract(
        self, 
        df: pd.DataFrame, 
        n_folds: int = 5
    ) -> list:
        """
        Extracts the folds from the input DataFrame.
        
        Parameters:
            - df (pd.DataFrame): The input DataFrame to be split into folds.
            - n_folds (int): The number of folds to create.
        
        Returns:
            - folds (list): A list of DataFrames representing the folds.
        """
        # ======= I. Create the folds =======
        folds = self.create_folds(df, n_folds=n_folds)
        
        # ======= II. Create the balanced folds =======
        balanced_folds = self.balance_data(folds)
        
        return folds, balanced_folds


