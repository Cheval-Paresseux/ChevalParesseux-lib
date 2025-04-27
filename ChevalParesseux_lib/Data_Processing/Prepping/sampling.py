import numpy as np
import pandas as pd
from joblib import Parallel, delayed


#! ==================================================================================== #
#! ============================= Extracting Functions ================================= #
def daily_cumsumTargetBars(
    data: pd.DataFrame, 
    target_bars: int,
    column_name: str = "volume",
    window_bars_estimation: int = 10,
    new_cols_methods: str = "mean",
    grouping_column: str = "date",
    pre_threshold: int = 1000,
    aggregation_dict: dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "ts": ["first", "last"],
        "date": "first",
        "bid_open": "first",
        "ask_open": "first",
    },
    n_jobs: int = 1,
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
        dfs_list = get_groups_list(data=data, column_name=grouping_column)
    else:
        dfs_list = [data]
    
    # ======= II. Extract the thresholds =======
    thresholds = []
    for idx in range(0, len(dfs_list)):
        if idx < window_bars_estimation:
            previous_days = dfs_list[:idx]
            threshold = pre_threshold
        else:
            previous_days = dfs_list[idx - window_bars_estimation : idx]
            average = np.mean([day[column_name].sum() for day in previous_days])
            threshold = int(average / target_bars)
        thresholds.append(threshold)
    
    # ======= III. Process each DataFrame in parallel or sequentially =======
    if n_jobs == 1:
        resampled_dfs = [get_cumsum_resample(df=dfs_list[idx], column_name=column_name, threshold=thresholds[idx], new_cols_method=new_cols_methods, aggregation_dict=aggregation_dict) for idx in range(len(dfs_list))]
    else:
        resampled_dfs = Parallel(n_jobs=n_jobs)(
            delayed(get_cumsum_resample)(df=dfs_list[idx], column_name=column_name, threshold=thresholds[idx], new_cols_method=new_cols_methods, aggregation_dict=aggregation_dict)
            for idx in range(len(dfs_list))
    )

    return resampled_dfs

#*____________________________________________________________________________________ #
def daily_cumsumWeightedTargetBars(
    data: pd.DataFrame, 
    target_bars: int,
    column_name: str = "volume",
    weight_column_name: str = "close",
    window_bars_estimation: int = 10,
    new_cols_methods: str = "mean",
    grouping_column: str = "date",
    pre_threshold: int = 1000,
    aggregation_dict: dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "ts": ["first", "last"],
        "date": "first",
        "bid_open": "first",
        "ask_open": "first",
    },
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    This function performs a resampling of the DataFrame based on the cumulative sum of a specified column.
    It estimates the threshold for each DataFrame based on the average of previous days, the first day is set to the target_bars.
    Parameters:
        - data (pd.DataFrame) : The DataFrame to be resampled.
        - target_bars (int) : The target number of bars.
        - column_name (str) : The column name to calculate the cumulative sum. Default is "volume".
        - weight_column_name (str) : The column name for the weighting. Default is "close".
        - window_bars_estimation (int) : The number of previous days to consider for estimating the average. Default is 10.
        - new_cols_methods (str) : The method to aggregate additional columns. Default is "mean".
        - grouping_column (str) : The column name to group by. Default is "date".
        - n_jobs (int) : The number of parallel jobs to run. Default is 1.
    
    Returns:
        - Resampled DataFrame.
    """
    # ======= I. Group the DataFrame if Necessary =======
    if grouping_column is not None:
        dfs_list = get_groups_list(data=data, column_name=grouping_column)
    else:
        dfs_list = [data]
    
    # ======= II. Extract the thresholds =======
    thresholds = []
    for idx in range(0, len(dfs_list)):
        if idx < window_bars_estimation:
            previous_days = dfs_list[:idx]
            threshold = pre_threshold
        else:
            previous_days = dfs_list[idx - window_bars_estimation : idx]
            average = np.mean([(day[column_name] * day[weight_column_name]).sum() for day in previous_days])
            threshold = int(average / target_bars)
        thresholds.append(threshold)
    
    # ======= III. Process each DataFrame in parallel or sequentially =======
    if n_jobs == 1:
        resampled_dfs = [get_cumsumWeighted_resample(df=dfs_list[idx], column_name=column_name, weight_column_name=weight_column_name, threshold=thresholds[idx], new_cols_method=new_cols_methods, aggregation_dict=aggregation_dict) for idx in range(len(dfs_list))]
    else:
        resampled_dfs = Parallel(n_jobs=n_jobs)(
            delayed(get_cumsumWeighted_resample)(df=dfs_list[idx], column_name=column_name, weight_column_name=weight_column_name, threshold=thresholds[idx], new_cols_method=new_cols_methods, aggregation_dict=aggregation_dict)
            for idx in range(len(dfs_list))
    )

    return resampled_dfs

#*____________________________________________________________________________________ #
def daily_volBars(
    data: pd.DataFrame, 
    column_name: str = "close",
    vol_threshold: float = 0.0005,
    new_cols_methods: str = "mean",
    grouping_column: str = "date",
    aggregation_dict: dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "ts": ["first", "last"],
        "date": "first",
        "bid_open": "first",
        "ask_open": "first",
    },
    n_jobs: int = 1,
) -> pd.DataFrame:
    # ======= I. Group the DataFrame if Necessary =======
    if grouping_column is not None:
        dfs_list = get_groups_list(data=data, column_name=grouping_column)
    else:
        dfs_list = [data]
    
    # ======= III. Process each DataFrame in parallel or sequentially =======
    if n_jobs == 1:
        resampled_dfs = [get_volatility_resample(df=dfs_list[idx], column_name=column_name, threshold=vol_threshold, new_cols_method=new_cols_methods, aggregation_dict=aggregation_dict) for idx in range(len(dfs_list))]
    else:
        resampled_dfs = Parallel(n_jobs=n_jobs)(
            delayed(get_volatility_resample)(df=dfs_list[idx], column_name=column_name, threshold=vol_threshold, new_cols_method=new_cols_methods, aggregation_dict=aggregation_dict)
            for idx in range(len(dfs_list))
    )

    return resampled_dfs



#! ==================================================================================== #
#! ============================= Sampling Functions =================================== #
def get_cumsum_resample(
    df: pd.DataFrame, 
    column_name: str = "volume",
    threshold: int = 1000,
    new_cols_method: str = "mean",
    aggregation_dict: dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "ts": ["first", "last"],
        "date": "first",
        "bid_open": "first",
        "ask_open": "first",
    },
) -> pd.DataFrame:
    """
    This function resamples the DataFrame based on the cumulative sum of a specified column.
    Parameters:
        - df (pd.DataFrame) : The DataFrame to be resampled.
        - column_name (str) : The column name to calculate the cumulative sum. Default is "volume".
        - threshold (int) : The threshold for the cumulative sum. Default is 1000.
        - new_cols_method (str) : The method to aggregate additional columns. Default is "mean".
    Returns:
        - Resampled DataFrame.
    __________
        Notes: The columns "open", "high", "low", "close", "volume", "ts", "date", "bid_open", and "ask_open" are expected to be present in the DataFrame.
               The function will aggregate these columns based on the specified method.
    """
    # ======= I. Initialization =======
    auxiliary_df = df.copy()
    bars_indexes = []
    cumulative_sum = 0
    idx = 0
    
    # ======= II. Extract the Indexes for Bars =======
    for value in auxiliary_df[column_name]:
        # II.1 Update the cumulative sum
        cumulative_sum += value

        # II.2 Check if the cumulative sum exceeds the threshold
        if cumulative_sum >= threshold:
            bars_indexes.append(idx)
            idx += 1
            cumulative_sum = 0
        else:
            bars_indexes.append(idx)

    # ======= III. Resample the DataFrame based on the volume bars =======
    # III.1 Add the bars indexes to the DataFrame
    auxiliary_df["new_bars"] = bars_indexes

    # III.2 Define the aggregation dictionary
    agg_dict = aggregation_dict.copy()

    # III.3 Check for additional columns and add them to the aggregation dictionary
    additional_cols = set(auxiliary_df.columns) - set(agg_dict.keys()) - {"new_bars"}
    for col in additional_cols:
        agg_dict[col] = new_cols_method

    # III.4 Perform the aggregation
    auxiliary_df = auxiliary_df.groupby("new_bars").agg(agg_dict)
    auxiliary_df.columns = [key for key in aggregation_dict.keys()] + list(additional_cols)
    
    auxiliary_df.reset_index(drop=True, inplace=True)
    
    return auxiliary_df

#*____________________________________________________________________________________ #
def get_cumsumWeighted_resample(
    df: pd.DataFrame, 
    column_name: str = "volume",
    weight_column_name: str = "close",
    threshold: int = 1000,
    new_cols_method: str = "mean",
    aggregation_dict: dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "ts": ["first", "last"],
        "date": "first",
        "bid_open": "first",
        "ask_open": "first",
    },
) -> pd.DataFrame:
    """
    This function resamples the DataFrame based on the cumulative sum of a specified column multiplied by the price.
    Parameters:
        - df (pd.DataFrame) : The DataFrame to be resampled.
        - column_name (str) : The column name to calculate the cumulative sum. Default is "volume".
        - weight_column_name (str) : The column name for the weighting. Default is "close".
        - threshold (int) : The threshold for the cumulative sum. Default is 1000.
        - new_cols_method (str) : The method to aggregate additional columns. Default is "mean".
    Returns:
        - Resampled DataFrame.
    __________
        Notes: The columns "open", "high", "low", "close", "volume", "ts", "date", "bid_open", and "ask_open" are expected to be present in the DataFrame.
               The function will aggregate these columns based on the specified method.
    """
    # ======= I. Initialization =======
    auxiliary_df = df.copy()
    bars_indexes = []
    cumulative_sum = 0
    idx = 0
    
    # ======= II. Extract the Indexes for Bars =======
    for value, weight in zip(auxiliary_df[column_name], auxiliary_df[weight_column_name]):
        # II.1 Update the cumulative sum
        dollar = value * weight
        cumulative_sum += dollar

        # II.2 Check if the cumulative sum exceeds the threshold
        if cumulative_sum >= threshold:
            bars_indexes.append(idx)
            idx += 1
            cumulative_sum = 0
        else:
            bars_indexes.append(idx)

    # ======= III. Resample the DataFrame based on the volume bars =======
    # III.1 Add the bars indexes to the DataFrame
    auxiliary_df["new_bars"] = bars_indexes

    # III.2 Define the aggregation dictionary
    agg_dict = aggregation_dict.copy()

    # III.3 Check for additional columns and add them to the aggregation dictionary
    additional_cols = set(auxiliary_df.columns) - set(agg_dict.keys()) - {"new_bars"}
    for col in additional_cols:
        agg_dict[col] = new_cols_method

    # III.4 Perform the aggregation
    auxiliary_df = auxiliary_df.groupby("new_bars").agg(agg_dict)
    auxiliary_df.columns = [key for key in aggregation_dict.keys()] + list(additional_cols)
    
    auxiliary_df.reset_index(drop=True, inplace=True)
    
    return auxiliary_df

#*____________________________________________________________________________________ #
def get_volatility_resample(
    df: pd.DataFrame, 
    column_name: str = "close",
    threshold: int = 0.001,
    new_cols_method: str = "mean",
    aggregation_dict: dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "ts": ["first", "last"],
        "date": "first",
        "bid_open": "first",
        "ask_open": "first",
    },
) -> pd.DataFrame:
    """
    This function resamples the DataFrame based on the volatility of a specified column.
    
    Parameters:
        - df (pd.DataFrame) : The DataFrame to be resampled.
        - column_name (str) : The column name to calculate the volatility. Default is "close".
        - threshold (int) : The threshold for the volatility. Default is 0.001.
        - new_cols_method (str) : The method to aggregate additional columns. Default is "mean".
    
    Returns:
        - Resampled DataFrame.
    __________
        Notes: The columns "open", "high", "low", "close", "volume", "ts", "date", "bid_open", and "ask_open" are expected to be present in the DataFrame.
               The function will aggregate these columns based on the specified method.
    """
    # ======= I. Initialization =======
    auxiliary_df = df.copy()
    bars_indexes = []
    idx = 0
    
    # ======= II. Extract the Indexes for Bars =======
    last_value = auxiliary_df[column_name].iloc[0]
    for value in auxiliary_df[column_name]:
        # II.1 Update the cumulative sum
        log_ret = np.abs(np.log(last_value / value))

        # II.2 Check if the cumulative sum exceeds the threshold
        if log_ret >= threshold:
            bars_indexes.append(idx)
            idx += 1
            last_value = value
        else:
            bars_indexes.append(idx)

    # ======= III. Resample the DataFrame based on the volume bars =======
    # III.1 Add the bars indexes to the DataFrame
    auxiliary_df["new_bars"] = bars_indexes

    # III.2 Define the aggregation dictionary
    agg_dict = aggregation_dict.copy()

    # III.3 Check for additional columns and add them to the aggregation dictionary
    additional_cols = set(auxiliary_df.columns) - set(agg_dict.keys()) - {"new_bars"}
    for col in additional_cols:
        agg_dict[col] = new_cols_method

    # III.4 Perform the aggregation
    auxiliary_df = auxiliary_df.groupby("new_bars").agg(agg_dict)
    auxiliary_df.columns = [key for key in aggregation_dict.keys()] + list(additional_cols)
    
    auxiliary_df.reset_index(drop=True, inplace=True)
    
    return auxiliary_df


#! ==================================================================================== #
#! ============================= Helper Functions =================================== #
def get_groups_list(
    data: pd.DataFrame, 
    column_name: str = "date"
) -> list:
    """
    This function groups the DataFrame by the specified column name and returns a list of DataFrames, each representing a group.
    Parameters:
        - data (pd.DataFrame) : The DataFrame to be grouped.
        - column_name (str) : The column name to group by. Default is "date".
    Returns:
        - List of DataFrames, each representing a group.
    """
    # ======= I. Group the DataFrame by the specified column =======
    df_grouped = data.groupby(column_name)
    
    # ======= II. Unpack the grouped_df into a list, each element is a DataFrame for one day =======
    dfs_list = [df_grouped.get_group(x) for x in df_grouped.groups]

    return dfs_list
