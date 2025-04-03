from ..Tuning import common as com
from ...utils import classification_metrics as metrics

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

#! ==================================================================================== #
#! ================================= Main Function ==================================== #
def get_samples_weights(
    labels_series: pd.Series, 
    price_series: pd.Series, 
    vol_window: int, 
    upper_barrier: float, 
    vertical_barrier: int,
    n_jobs: int = 1,
) -> pd.DataFrame:
    # ======= I. Prepare the Auxiliary DataFrame =======
    labels = labels_series.dropna().copy()
    price = price_series.loc[labels.index].copy()
    rolling_vol = price_series.pct_change().rolling(vol_window).std() * np.sqrt(vol_window)
    rolling_vol.rename('volatility', inplace=True)

    auxiliary_df = pd.concat([labels, price, rolling_vol], axis=1).dropna()
    
    # ======= II. Extract Events =======
    events = Parallel(n_jobs=n_jobs)(
        delayed(extract_event)(idx, row, auxiliary_df, upper_barrier, vertical_barrier)
        for idx, row in auxiliary_df.iterrows()
    )
    auxiliary_df['start_event'] = [event[0] for event in events]
    auxiliary_df['end_event'] = [event[1] for event in events]
    auxiliary_df['event_returns'] = [event[2] for event in events]
    
    # ======= III. Compute the Average Uniqueness =======
    concurrent_events = auxiliary_df.apply(lambda row: count_concurrent_events(row, auxiliary_df), axis=1)
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


#! ==================================================================================== #
#! ================================= Helper Function ==================================== #
def extract_event(idx, row, df, upper_barrier, vertical_barrier):
    # ======= I. Extract close and barrier =======
    current_close = row['close']
    barrier = row['volatility'] * upper_barrier
    
    # ==
    if row['labels'] == 1:
        target_close = current_close * (1 + barrier)
        barrier_cross = df[(df['close'] >= target_close) & (df.index > idx)]

    elif row['labels'] == -1:
        target_close = current_close * (1 - barrier)
        barrier_cross = df[(df['close'] <= target_close) & (df.index > idx)]
        
    else:
        event_returns = (1 + barrier) / 2
        return idx, idx + vertical_barrier, event_returns
    
    barrier_hit_idx = barrier_cross.index.min() if not barrier_cross.empty else idx + vertical_barrier
    if barrier_hit_idx < idx + vertical_barrier:
        hit_close = df.loc[barrier_hit_idx, 'close']
        event_returns = np.abs(np.log(hit_close / current_close))
    else:
        event_returns = np.nan
        
    return idx, barrier_hit_idx, event_returns


#*____________________________________________________________________________________ #
def count_concurrent_events(row, df):
    label = row['labels']
    start_idx = row['start_event']
    end_idx = row['end_event']
    
    mask_prev = (df['labels'] == label) & (df['start_event'] < start_idx) & (df['end_event'] >= start_idx)
    mask_next = (df['labels'] == label) & (df['start_event'] < end_idx) & (df['end_event'] >= end_idx)
    
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
