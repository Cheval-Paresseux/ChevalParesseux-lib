import pandas as pd 
import numpy as np

#! ==================================================================================== #
#! ============================= Sampling Functions =================================== #
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

#*____________________________________________________________________________________ #
def get_cumsum_resample(
    df: pd.DataFrame, 
    column_name: str = "volume",
    threshold: int = 1000,
    new_cols_method: str = "mean"
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
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "ts": ["first", "last"],
        "date": "first",
        "bid_open": "first",
        "ask_open": "first",
    }

    # III.3 Check for additional columns and add them to the aggregation dictionary
    additional_cols = set(auxiliary_df.columns) - set(agg_dict.keys()) - {"new_bars"}
    for col in additional_cols:
        agg_dict[col] = new_cols_method

    # III.4 Perform the aggregation
    auxiliary_df = auxiliary_df.groupby("new_bars").agg(agg_dict)
    auxiliary_df.columns = [ "open", "high", "low", "close", "volume", "ts_open", "ts_close", "date", "bid_open", "ask_open"] + list(additional_cols)
    
    auxiliary_df.reset_index(drop=True, inplace=True)
    
    return auxiliary_df

