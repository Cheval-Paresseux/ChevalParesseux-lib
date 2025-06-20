from ..sampling import common as com

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Union, Self, Optional



#! ==================================================================================== #
#! ================================= Main Function ==================================== #
class Temporal_uniqueness_selection(com.DatasetBuilder):
    """
    Resampling method for temporal uniqueness selection.
    
    This class is used to extract a new dataset from the original one based on random drawing of samples.
    Each label is mapped to a probability that depends on the average uniqueness of the event, the time decay, and the event returns.
    """
    #?_____________________________ Initialization methods _______________________________ #
    def __init__(
        self, 
        n_jobs: int = 1, 
        random_state: int = 72
    ):
        """
        Constructor for the temporal_uniqueness_selection class.
        
        Parameters:
            - n_jobs (int): The number of jobs to run in parallel. Default is 1 (no parallelization).
            - random_state (int): The random state for reproducibility. Default is 72.
        """
        super().__init__(n_jobs=n_jobs)
        self.random_state = random_state
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        label_column: list = ['label'],
        price_column: list = ['close'],
        n_samples: list = [1.3],
        replacement: list = [True],
        balancing: list = [True],
        vol_window: list = [21],
        upper_barrier: list = [0.5],
        vertical_barrier: list = [21],
        grouping_column: Optional[list] = None,
    ) -> Self:
        """
        Set the parameters for the temporal uniqueness selection.
        
        Parameters:
            - label_column (list): The column names for the labels. Default is ['label'].
            - price_column (list): The column names for the prices. Default is ['close'].
            - n_samples (list): The number of samples to extract in percentage according to orignial df size. Default is [1.3].
            - replacement (list): Whether to sample with replacement. Default is [False].
            - balancing (list): Whether to balance the dataset. Default is [False].
            - vol_window (list): The window size for calculating the rolling volatility. Default is [10].
            - upper_barrier (list): The upper barrier for the event. Default is [0.5].
            - vertical_barrier (list): The vertical barrier for the event. Default is [20].
        
        Returns:
            - self: The instance of the class with the updated parameters.
        """
        self.params = {
            'label_column': label_column,
            'price_column': price_column,
            'n_samples': n_samples,
            'replacement': replacement,
            'balancing': balancing,
            'vol_window': vol_window,
            'upper_barrier': upper_barrier,
            'vertical_barrier': vertical_barrier,
            'grouping_column': grouping_column,
        }
        
        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self,
        data: Union[pd.DataFrame, list],
        grouping_column: str,
    ) -> list:
        """
        This function groups the DataFrame by the specified column name and returns a list of DataFrames, each representing a group.
        
        Parameters:
            - data (pd.DataFrame) : The DataFrame to be grouped.
            - grouping_column (str) : The column name to group by.
        
        Returns:
            - List of DataFrames, each representing a group.
        """
        # ======= 0. Define grouping method =======
        def groupby_method(
            df: pd.DataFrame, 
            grouping_column: str
        ) -> list:
            """
            Groups the DataFrame by the specified column name and returns a list of DataFrames, each representing a group.
            
            Parameters:
                - df (pd.DataFrame) : The DataFrame to be grouped.
                - grouping_column (str) : The column name to group by.
            
            Returns:
                - List of DataFrames, each representing a group.
            """
            if grouping_column is not None:
                df_grouped = df.groupby(grouping_column)
                dfs_list = [df_grouped.get_group(x) for x in df_grouped.groups]
                dfs_list = [df.reset_index(drop=True) for df in dfs_list]
            
            else:
                dfs_list = [df.reset_index(drop=True)]
            
            return dfs_list
        
        # ======= 1. Apply grouping =======
        if isinstance(data, list):
            processed_data = []
            for df in data:
                dfs_list = groupby_method(df, grouping_column)
                processed_data.extend(dfs_list)    
        else:
            processed_data = groupby_method(data, grouping_column)

        return processed_data
    
    #?________________________________ Auxiliary methods _________________________________ #
    def extract_event(
        self,
        row: pd.Series, 
        df: pd.DataFrame, 
        upper_barrier: float, 
        vertical_barrier: int, 
        label_column: str, 
        price_column: str
    ) -> tuple:
        """
        Extracts the event information from the current row of the DataFrame.
        
        Returns:
            - tuple: (start_index, end_index, event_returns)
        """
        idx = row.name
        current_price = row[price_column]
        barrier = row['volatility'] * upper_barrier

        # Convert idx (label) to position for slicing
        idx_pos = df.index.get_loc(idx)
        max_pos = min(idx_pos + vertical_barrier, len(df) - 1)
        futur = df.iloc[idx_pos : max_pos + 1].copy() 

        max_price = futur[price_column].max()
        min_price = futur[price_column].min()
        max_price_index = futur[price_column].idxmax()
        min_price_index = futur[price_column].idxmin()

        max_return = (max_price - current_price) / current_price
        min_return = (min_price - current_price) / current_price

        # ======= II. Define the event =======
        if row[label_column] == 1:
            if max_return < barrier:
                barrier_hit_idx = futur.index[-1]
                final_return = (futur[price_column].iloc[-1] / current_price) - 1
                event_returns = final_return if final_return > 0 else np.nan
            else:
                barrier_hit_idx = max_price_index
                event_returns = max_return

        elif row[label_column] == -1:
            if min_return > -barrier:
                barrier_hit_idx = futur.index[-1]
                final_return = (futur[price_column].iloc[-1] / current_price) - 1
                event_returns = -final_return if final_return < 0 else np.nan
            else:
                barrier_hit_idx = min_price_index
                event_returns = -min_return

        else:
            barrier_hit_idx = futur.index[-1]
            event_returns = barrier / 2  # This is arbitrary but OK for neutral labels

        return idx, barrier_hit_idx, event_returns

    #?____________________________________________________________________________________ #
    def count_concurrent_events(
        self,
        row: pd.Series, 
        df: pd.DataFrame, 
        label_column: str
    ) -> int:
        """
        Counts the number of concurrent events for the current row in the DataFrame.
        
        Parameters:
            - row (pd.Series): The current row of the DataFrame.
            - df (pd.DataFrame): The DataFrame containing the data.
            - label_column (str): The name of the labels column.

        Returns:
            - int: The number of concurrent events.
        """
        # ======= I. Get the label and event indices =======
        label = row[label_column]
        start_idx = row['start_event']
        end_idx = row['end_event']
        
        # ======= II. Get the concurrent events =======
        mask_prev = (df[label_column] == label) & (df['start_event'] < start_idx) & (df['end_event'] >= start_idx)
        mask_next = (df[label_column] == label) & (df['start_event'] < end_idx) & (df['end_event'] >= end_idx)
        
        # ======= III. Count the concurrent events =======
        nb_concurrents_events_prev = df[mask_prev].shape[0]
        nb_concurrents_events_next = df[mask_next].shape[0]
        
        nb_concurrents_events = nb_concurrents_events_prev + nb_concurrents_events_next + 1
        
        return nb_concurrents_events

    #?____________________________________________________________________________________ #
    def get_linear_Tdecay(
        self,
        series: pd.Series
    ) -> pd.Series:
        """
        Computes the linear time decay for a given series.
        
        Parameters:
            - series (pd.Series): The input series for which to compute the time decay.
        
        Returns:
            - pd.Series: The series containing the time decay values.
        """
        # ======= I. Compute the weights =======
        n = len(series)
        weights = np.linspace(1/n, 1, n)
        
        # ======= II. Transform into series =======
        time_decay = pd.Series(weights, index=series.index)
        
        return time_decay
    
    #?____________________________________________________________________________________ #
    def get_samples_weights(
        self,
        label_series: pd.Series, 
        price_series: pd.Series, 
        vol_window: int, 
        upper_barrier: float, 
        vertical_barrier: int,
        label_column: str,
        price_column: str,
    ) -> pd.DataFrame:
        """
        This function computes the sample weights for each event in the dataset, based on the average uniqueness,
        time decay, and event returns. The sample weights are used to balance the dataset for training a model.
        
        Parameters:
            - label_series (pd.Series): The series containing the labels for the events.
            - price_series (pd.Series): The series containing the price data.
            - vol_window (int): The window size for calculating the rolling volatility.
            - upper_barrier (float): The upper barrier for the event.
            - vertical_barrier (int): The vertical barrier for the event.
            - label_column (str): The name of the labels column. 
            - price_column (str): The name of the price column.
        
        Returns:
            - sample_weights (pd.Series): The series containing the sample weights for each event.
            - auxiliary_df (pd.DataFrame): The DataFrame containing the auxiliary data for the events.
        """
        # ======= I. Prepare the Auxiliary DataFrame =======
        labels = label_series.dropna().copy()
        price = price_series.loc[labels.index].copy()
        rolling_vol = price_series.pct_change().rolling(vol_window).std() * np.sqrt(vol_window)
        rolling_vol.rename('volatility', inplace=True)

        auxiliary_df = pd.concat([labels, price, rolling_vol], axis=1).dropna()
        
        # ======= II. Extract Events =======
        events = Parallel(n_jobs=self.n_jobs)(
            delayed(self.extract_event)(row, auxiliary_df, upper_barrier, vertical_barrier, label_column, price_column)
            for _, row in auxiliary_df.iterrows()
        )

        auxiliary_df['start_event'] = [event[0] for event in events]
        auxiliary_df['end_event'] = [event[1] for event in events]
        auxiliary_df['event_returns'] = [event[2] for event in events]
        
        # ======= III. Compute the Average Uniqueness =======
        concurrent_events = auxiliary_df.apply(lambda row: self.count_concurrent_events(row, auxiliary_df, label_column), axis=1)
        auxiliary_df['average_uniqueness'] = 1 / concurrent_events
        auxiliary_df['time_decay'] = self.get_linear_Tdecay(auxiliary_df['event_returns'])
        
        # ======= IV. Compute the Sample Weights =======
        auxiliary_df['event_returns'] /= auxiliary_df['event_returns'].sum()
        auxiliary_df['average_uniqueness'] /= auxiliary_df['average_uniqueness'].sum()
        auxiliary_df['time_decay'] /= auxiliary_df['time_decay'].sum()
        
        auxiliary_df['sample_weights'] = auxiliary_df['average_uniqueness'] * auxiliary_df['time_decay'] * auxiliary_df['event_returns']
        auxiliary_df['sample_weights'] /= auxiliary_df['sample_weights'].sum()
        
        sample_weights = auxiliary_df['sample_weights'].copy()
        
        # ======= Ensure Same Index as labels_series =======
        sample_weights = sample_weights.reindex(label_series.index)
        
        return sample_weights, auxiliary_df
    
    #?____________________________________________________________________________________ #
    def get_dataset(
        self, 
        data: Union[pd.DataFrame, list],
        grouping_column: str,
        label_column: str,
        price_column: str,
        vol_window: int,
        upper_barrier: float,
        vertical_barrier: int,
        n_samples: float,
        replacement: bool,
        balancing: bool,
    ) -> pd.DataFrame:
        """
        Extract a new dataset from the original one based on random drawing of samples, probabilities computed using the labels weights.
        
        Parameters:
            - data (pd.DataFrame or list): The input data to be processed.
            - grouping_column (str): The column name to group by.
            - label_column (str): The column name for the labels.
            - price_column (str): The column name for the prices.
            - vol_window (int): The window size for calculating the rolling volatility.
            - upper_barrier (float): The upper barrier for the event.
            - vertical_barrier (int): The vertical barrier for the event.
            - n_samples (float): The number of samples to extract in percentage according to original DataFrame size.
            - replacement (bool): Whether to sample with replacement.
            - balancing (bool): Whether to balance the dataset.
        
        Returns:
            - results (list): A list of DataFrames, each representing a sampled dataset.
        """
        # ======= I. Process Data =======
        np.random.seed(self.random_state)
        processed_data = self.process_data(data=data, grouping_column=grouping_column)
        
        # ======= II. Create the DataFrames =======
        results = []
        for df in processed_data:
            # ----- 1. Extract the sample weights -----
            training_df = df.copy()
            training_df.dropna(inplace=True)
            
            sample_weights, _ = self.get_samples_weights(
                label_series=training_df[label_column], 
                price_series=training_df[price_column], 
                vol_window=vol_window, 
                upper_barrier=upper_barrier, 
                vertical_barrier=vertical_barrier,
                label_column=label_column,
                price_column=price_column,
            )
            training_df = pd.concat([training_df, sample_weights], axis=1)
            training_df = training_df.dropna(axis=0)

            # ----- 2. If rebalancing, extract for each label -----
            if balancing:
                # 2.1 Get the number of labels
                available_labels = training_df[label_column].unique()
                labels_specific_dfs = [training_df[training_df[label_column] == label].copy() for label in available_labels]
                nb_labels = len(labels_specific_dfs)
                
                # 2.2 For each label, extract the resampled dataframe
                sampled_dfs = []
                for unique_df in labels_specific_dfs:
                    # 2.2.1 Normalize the sample weights to be a probability
                    unique_df['sample_weights'] = unique_df['sample_weights'] / unique_df['sample_weights'].sum()
                    
                    # 2.2.2 Adjust the number of samples if replacement
                    sample_size = training_df.shape[0] * n_samples
                    target_nb_samples = int(sample_size // nb_labels)
                    if not replacement:
                        nb_samples = min(target_nb_samples, unique_df.shape[0])
                    else:
                        nb_samples = target_nb_samples
                    
                    # 2.2.3 Sample the indices
                    sampled_indices = np.random.choice(unique_df.index, size=nb_samples, replace=replacement, p=unique_df["sample_weights"])
                    df_sampled = unique_df.loc[sampled_indices].reset_index(drop=True)
                    sampled_dfs.append(df_sampled)
            
                # 2.3 Concatenate the sampled DataFrames
                df_sampled = pd.concat(sampled_dfs, axis=0).reset_index(drop=True)
                df_sampled = df_sampled.drop(columns=['sample_weights'])
                results.append(df_sampled)
            
            # ----- 3. Else, extract directly using label weight -----
            else:
                # 3.1 Normalize the sample weights to be a probability
                training_df['sample_weights'] = training_df['sample_weights'] / training_df['sample_weights'].sum()
                
                # 3.2 Adjust the number of samples if replacement
                if not replacement:
                    n_samples = min(n_samples, training_df.shape[0])
                    print(f'n_samples: {n_samples}')
                
                # 3.3 Sample the indices
                sampled_indices = np.random.choice(training_df.index, size=n_samples, replace=replacement, p=training_df["sample_weights"])
                df_sampled = training_df.loc[sampled_indices].reset_index(drop=True)
                df_sampled = df_sampled.drop(columns=['sample_weights'])
                results.append(df_sampled)
        
        return results

