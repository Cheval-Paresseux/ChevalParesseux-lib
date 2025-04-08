import pandas as pd
import numpy as np
from joblib import Parallel, delayed

#! ==================================================================================== #
#! ================================= Main Function ==================================== #
class SetGenerator:
    def __init__(
        self, 
        training_df: pd.DataFrame, 
        n_jobs: int = 1, 
        random_state: int = 72
    ):
        # ===== I. Initialize the Inputs =======
        self.training_df = training_df.copy()
        
        # ===== II. Initialize the Attributes =======
        self.balanced_dfs = None
        self.folds = None
        
        # ===== III. Initialize the Parameters =======
        self.params = None
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        labels_name: str = 'labels',
        price_name: str = 'close',
        n_samples: int = 1000,
        replacement: bool = False,
        vol_window: int = 10,
        upper_barrier: float = 0.5,
        vertical_barrier: int = 20,
    ) -> None:
        """
        This function sets the parameters for the SetGenerator class.
        Parameters:
            - labels_name (str): The name of the column containing the labels.
            - price_name (str): The name of the column containing the price data.
            - n_samples (int): The number of samples to generate.
            - replacement (bool): Whether to sample with replacement or not.
            - vol_window (int): The window size for calculating the rolling volatility.
            - upper_barrier (float): The upper barrier for the event.
            - vertical_barrier (int): The vertical barrier for the event.
        """
        self.params = {
            'labels_name': labels_name,
            'price_name': price_name,
            'n_samples': n_samples,
            'replacement': replacement,
            'vol_window': vol_window,
            'upper_barrier': upper_barrier,
            'vertical_barrier': vertical_barrier,
        }
        
        return self
        
    #?____________________________________________________________________________________ #
    def balance_data(self, df_list: list) -> pd.DataFrame:
        """
        This function balances the data by creating random samples of each class.
        Parameters:
            - df_list (list): A list of DataFrames to be balanced.
        Returns:
            - balanced_dfs (list): A list of balanced DataFrames.
        """
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
    def create_folds(self, df: pd.DataFrame, n_folds: int = 5) -> list:
        """
        This function creates folds for cross-validation.
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


#*____________________________________________________________________________________ #
def create_random_sample(
    df: pd.DataFrame, 
    n_samples: int = 1000,
    replacement: bool = False,
    vol_window: int = 10, 
    upper_barrier: float = 0.5, 
    vertical_barrier: int = 20,
    labels_name: str = 'labels',
    price_name: str = 'close',
    n_jobs: int = 1, 
    random_state: int = 72
) -> pd.DataFrame:
    """
    This function creates a random sample of the input DataFrame based on the sample weights.
    Parameters:
        - df (pd.DataFrame): The input DataFrame containing the data.
        - n_samples (int): The number of samples to generate.
        - replacement (bool): Whether to sample with replacement or not.
        - vol_window (int): The window size for calculating the rolling volatility.
        - upper_barrier (float): The upper barrier for the event.
        - vertical_barrier (int): The vertical barrier for the event.
        - labels_name (str): The name of the column containing the labels.
        - price_name (str): The name of the column containing the price data.
        - n_jobs (int): The number of jobs to run in parallel. Default is 1 (no parallelization).
    
    Returns:
        - df_sampled (pd.DataFrame): The sampled DataFrame.
    """
    # ======= I. Get the Input DataFrame =======
    np.random.seed(random_state)
    
    training_df = df.copy()
    training_df.dropna(inplace=True)
    
    # ======= II. Extract the sample weights =======
    sample_weights, _ = get_samples_weights(
        labels_series=training_df[labels_name], 
        price_series=training_df[price_name], 
        vol_window=vol_window, 
        upper_barrier=upper_barrier, 
        vertical_barrier=vertical_barrier,
        labels_name=labels_name,
        price_name=price_name,
        n_jobs=n_jobs
    )
    training_df = pd.concat([training_df, sample_weights], axis=1)
    training_df = training_df.dropna(axis=0)
    
    # ======= III. Create the Random Sample =======
    sampled_indices = np.random.choice(training_df.index, size=n_samples, replace=replacement, p=training_df["sample_weights"])
    df_sampled = training_df.loc[sampled_indices].reset_index(drop=True)
    
    return df_sampled


#! ==================================================================================== #
#! ================================= Helper Function ================================== #
def get_samples_weights(
    labels_series: pd.Series, 
    price_series: pd.Series, 
    vol_window: int, 
    upper_barrier: float, 
    vertical_barrier: int,
    labels_name: str = 'labels',
    price_name: str = 'close',
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    This funciton computes the sample weights for each event in the dataset, based on the average uniqueness,
    time decay, and event returns. The sample weights are used to balance the dataset for training a model.
    Parameters:
        - labels_series (pd.Series): The Series containing the labels for each event.
        - price_series (pd.Series): The Series containing the price data for each event.
        - vol_window (int): The window size for calculating the rolling volatility.
        - upper_barrier (float): The upper barrier for the event.
        - vertical_barrier (int): The vertical barrier for the event.
        - n_jobs (int): The number of jobs to run in parallel. Default is 1 (no parallelization).
    
    Returns: 
        - sample_weights (pd.Series): The Series containing the sample weights for each event.
        - auxiliary_df (pd.DataFrame): The DataFrame containing the auxiliary data for each event.
    """
    # ======= I. Prepare the Auxiliary DataFrame =======
    labels = labels_series.dropna().copy()
    price = price_series.loc[labels.index].copy()
    rolling_vol = price_series.pct_change().rolling(vol_window).std() * np.sqrt(vol_window)
    rolling_vol.rename('volatility', inplace=True)

    auxiliary_df = pd.concat([labels, price, rolling_vol], axis=1).dropna()
    
    # ======= II. Extract Events =======
    events = Parallel(n_jobs=n_jobs)(
        delayed(extract_event)(idx, row, auxiliary_df, upper_barrier, vertical_barrier, labels_name, price_name)
        for idx, row in auxiliary_df.iterrows()
    )
    auxiliary_df['start_event'] = [event[0] for event in events]
    auxiliary_df['end_event'] = [event[1] for event in events]
    auxiliary_df['event_returns'] = [event[2] for event in events]
    
    # ======= III. Compute the Average Uniqueness =======
    concurrent_events = auxiliary_df.apply(lambda row: count_concurrent_events(row, auxiliary_df, labels_name), axis=1)
    auxiliary_df['average_uniqueness'] = 1 / concurrent_events
    auxiliary_df['time_decay'] = get_linear_Tdecay(auxiliary_df['event_returns'])
    
    # ======= IV. Compute the Sample Weights =======
    auxiliary_df['event_returns'] /= auxiliary_df['event_returns'].sum()
    auxiliary_df['average_uniqueness'] /= auxiliary_df['average_uniqueness'].sum()
    auxiliary_df['time_decay'] /= auxiliary_df['time_decay'].sum()
    
    auxiliary_df['sample_weights'] = auxiliary_df['average_uniqueness'] * auxiliary_df['time_decay'] * auxiliary_df['event_returns']
    auxiliary_df['sample_weights'] /= auxiliary_df['sample_weights'].sum()
    
    sample_weights = auxiliary_df['sample_weights'].copy()
    
    # ======= Ensure Same Index as labels_series =======
    sample_weights = sample_weights.reindex(labels_series.index)
    
    return sample_weights, auxiliary_df

#*____________________________________________________________________________________ #
def extract_event(idx, row, df, upper_barrier, vertical_barrier, labels_name, price_name):
    # ======= I. Extract close and barrier =======
    current_close = row[price_name]
    barrier = row['volatility'] * upper_barrier
    
    # ======= II. Define the event =======
    if row[labels_name] == 1:
        target_close = current_close * (1 + barrier)
        barrier_cross = df[(df[price_name] >= target_close) & (df.index > idx)]

    elif row[labels_name] == -1:
        target_close = current_close * (1 - barrier)
        barrier_cross = df[(df[price_name] <= target_close) & (df.index > idx)]
        
    else:
        event_returns = (1 + barrier) / 2
        return idx, idx + vertical_barrier, event_returns
    
    # ======= III. Get the event start and end =======
    barrier_hit_idx = barrier_cross.index.min() if not barrier_cross.empty else idx + vertical_barrier
    
    # ======= IV. Get the event returns =======
    if barrier_hit_idx < idx + vertical_barrier:
        hit_close = df.loc[barrier_hit_idx, price_name]
        event_returns = np.abs(np.log(hit_close / current_close))
    else:
        event_returns = np.nan
        
    return idx, barrier_hit_idx, event_returns

#*____________________________________________________________________________________ #
def count_concurrent_events(row, df, labels_name):
    label = row[labels_name]
    start_idx = row['start_event']
    end_idx = row['end_event']
    
    mask_prev = (df[labels_name] == label) & (df['start_event'] < start_idx) & (df['end_event'] >= start_idx)
    mask_next = (df[labels_name] == label) & (df['start_event'] < end_idx) & (df['end_event'] >= end_idx)
    
    nb_concurrents_events_prev = df[mask_prev].shape[0]
    nb_concurrents_events_next = df[mask_next].shape[0]
    
    nb_concurrents_events = nb_concurrents_events_prev + nb_concurrents_events_next + 1
    
    return nb_concurrents_events

#*____________________________________________________________________________________ #
def get_linear_Tdecay(series: pd.Series) -> pd.Series:
    n = len(series)
    weights = np.linspace(1/n, 1, n)
    
    time_decay = pd.Series(weights, index=series.index)
    
    return time_decay
