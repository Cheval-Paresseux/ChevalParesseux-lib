from . import common as com

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


#! ==================================================================================== #
#! ============================= Sampling Functions =================================== #
def extract_targetBars(
    data: pd.DataFrame, 
    target_bars: int,
    column_name: str = "volume",
    window_bars_estimation: int = 10,
    new_cols_methods: str = "mean",
    grouping_column: str = "date",
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    This function performs a resampling of the DataFrame based on the cumulative sum of a specified column.
    It estimates the threshold for each DataFrame based on the average of previous days, the first day is set to the target_bars.
    Parameters:
        - data (pd.DataFrame) : The DataFrame to be resampled.
        - target_bars (int) : The target number of bars.
        - column_name (str) : The column name to calculate the cumulative sum. Default is "volume".
        - window_bars_estimation (int) : The number of previous days to consider for estimating the average. Default is 10.
        - new_cols_methods (str) : The method to aggregate additional columns. Default is "mean".
        - grouping_column (str) : The column name to group by. Default is "date".
        - n_jobs (int) : The number of parallel jobs to run. Default is 1.
    
    Returns:
        - Resampled DataFrame.
    """
    # ======= I. Group the DataFrame if Necessary =======
    if grouping_column is not None:
        dfs_list = com.get_groups_list(data=data, column_name=grouping_column)
    else:
        dfs_list = [data]
    
    # ======= II. Extract the thresholds =======
    thresholds = [target_bars] # The first threshold is the target_bars as we don't have previous days to estimate the average.
    for idx in range(1, len(dfs_list)):
        if idx < window_bars_estimation:
            previous_days = dfs_list[:idx]
        else:
            previous_days = dfs_list[idx - window_bars_estimation : idx]

        average = np.mean([day[column_name].sum() for day in previous_days])
        threshold = int(average / target_bars)
        thresholds.append(threshold)
    
    # ======= III. Process each DataFrame in parallel or sequentially =======
    if n_jobs == 1:
        resampled_dfs = [com.get_cumsum_resample(df=dfs_list[idx], column_name=column_name, threshold=thresholds[idx], new_cols_method=new_cols_methods) for idx in range(len(dfs_list))]
    else:
        resampled_dfs = Parallel(n_jobs=n_jobs)(
            delayed(com.get_cumsum_resample)(df=dfs_list[idx], column_name=column_name, threshold=thresholds[idx], new_cols_method=new_cols_methods)
            for idx in range(len(dfs_list))
    )

    return resampled_dfs
