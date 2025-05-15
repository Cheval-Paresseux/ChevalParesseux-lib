from ..sampling import common as com

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Union, Self, Optional



#! ==================================================================================== #
#! =================================== Builders  ====================================== #
class Cumsum_bars(com.DatasetBuilder):
    """
    Class for building a dataset based on cumulative sum bars.
    
    This class inherits from the DatasetBuilder class and implements methods for
        - setting parameters for dataset extraction
        - grouping data given a column name
        - extracting cumulative sum bars from the data
    """
    #?_____________________________ Initialization methods _______________________________ #
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the DatasetBuilder object.

        Parameters:
            - n_jobs (int): Number of parallel jobs to use for computation.
        """
        # ======= I. Initialize Class =======
        super().__init__(n_jobs=n_jobs)
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        cumsum_column: list = ['volume','close'],
        aggregation_rules: list = [{"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "date": "first"}],
        general_aggregation_rules: list = ["mean"],
        grouping_column: Optional[list] = None,
        threshold: Optional[list] = None,
        weighting_column: Optional[list] = None,
        target_bars: Optional[list] = None,
        target_estimation_window: Optional[list] = None,
        target_pre_threshold: Optional[list] = None,
    ) -> Self:
        """
        Sets the parameter grid for the datset extraction.

        Parameters:
            - cumsum_column (list): The column name to calculate the cumulative sum. Default is ["volume", "close"].
            - aggregation_rules (list): Dictionary defining the aggregation rules. Default is [{"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "date": "first"}].
            - general_aggregation_rules (list): General aggregation rules to be applied. Default is ["mean"].
            - grouping_column (list): The column name to group by. Default is None.
            - threshold (list): The threshold value for filtering. Default is None.
            - weighting_column (list): The column name to be used for weighting. Default is None.
            - target_bars (list): The target number of bars for estimation. Default is None.
            - target_estimation_window (list): The target estimation window size. Default is None.
            - target_pre_threshold (list): The pre-threshold value for filtering. Default is None.

        Returns:
            - Self: The instance of the class with the parameter grid set.
        """
        # ======= I. Initialize Class =======
        self.params = {
            "cumsum_column": cumsum_column,
            "aggregation_rules": aggregation_rules,
            "general_aggregation_rules": general_aggregation_rules,
            "grouping_column": grouping_column,
            "threshold": threshold,
            "weighting_column": weighting_column,
            "target_bars": target_bars,
            "target_estimation_window": target_estimation_window,
            "target_pre_threshold": target_pre_threshold
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
            
            else:
                dfs_list = [df]
            
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
    def get_cumsum_resample(
        self,
        data: pd.DataFrame, 
        cumsum_column: str,
        aggregation_rules: dict,
        general_aggregation_rules: str,
        threshold: Optional[float] = None,
        weighting_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        This function resamples the DataFrame based on the cumulative sum of a specified column.
        
        Parameters:
            - data (pd.DataFrame) : The DataFrame to be resampled.
            - cumsum_column (str) : The column name to calculate the cumulative sum.
            - aggregation_rules (dict) : Dictionary defining the aggregation rules.
            - general_aggregation_rules (str) : General aggregation rules to be applied.
            - threshold (float) : The threshold value for filtering.
            - weighting_column (str) : The column name to be used for weighting.
        
        Returns:
            - pd.DataFrame : The resampled DataFrame.
        """
        # ======= I. Initialization =======
        auxiliary_df = data.copy()
        bars_indexes = []
        cumulative_sum = 0
        idx = 0
        
        # ======= II. Extract the Indexes for Bars =======
        for value in auxiliary_df[cumsum_column]:
            # II.1 Update the cumulative sum
            if weighting_column is not None:
                current_value = value * auxiliary_df[weighting_column].iloc[idx]
            else:
                current_value = value
            cumulative_sum += current_value

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
        agg_dict = aggregation_rules.copy()

        # III.3 Check for additional columns and add them to the aggregation dictionary
        additional_cols = set(auxiliary_df.columns) - set(agg_dict.keys()) - {"new_bars"}
        for col in additional_cols:
            agg_dict[col] = general_aggregation_rules

        # III.4 Perform the aggregation 
        auxiliary_df = auxiliary_df.groupby("new_bars").agg(agg_dict)
        # auxiliary_df.columns = [key for key in aggregation_rules.keys()] + list(additional_cols)
        auxiliary_df = auxiliary_df[list(aggregation_rules.keys()) + sorted(additional_cols)]
        
        auxiliary_df.reset_index(drop=True, inplace=True)
        
        return auxiliary_df

    #?____________________________________________________________________________________ #
    def get_dataset(
        self, 
        data: Union[pd.DataFrame, list],
        cumsum_column: str,
        aggregation_rules: dict,
        general_aggregation_rules: str,
        grouping_column: Optional[str] = None,
        threshold: Optional[float] = None,
        weighting_column: Optional[str] = None,
        target_bars: Optional[int] = None,
        target_estimation_window: Optional[int] = None,
        target_pre_threshold: Optional[float] = None,
    ):
        """
        This function extracts the dataset based on the specified parameters.
        
        Parameters:
            - data (Union[pd.DataFrame, list]) : The DataFrame to be processed.
            - cumsum_column (str) : The column name to calculate the cumulative sum.
            - aggregation_rules (dict) : Dictionary defining the aggregation rules.
            - general_aggregation_rules (str) : General aggregation rules to be applied.
            - grouping_column (str) : The column name to group by.
            - threshold (float) : The threshold value for filtering.
            - weighting_column (str) : The column name to be used for weighting.
            - target_bars (int) : The target number of bars for estimation.
            - target_estimation_window (int) : The target estimation window size.
            - target_pre_threshold (float) : The pre-threshold value for filtering.
        
        Returns:
            - List of DataFrames, each representing a group.
        """
        # ======= I. Process Data =======
        processed_data = self.process_data(data, grouping_column=grouping_column)

        # ======= II. Extract Thresholds Values =======
        if target_bars is not None:
            thresholds = []
            for idx in range(0, len(processed_data)):
                if idx < target_estimation_window:
                    threshold_value = target_pre_threshold
                else:
                    previous_days = processed_data[idx - target_estimation_window : idx]
                    average = np.mean([day[cumsum_column].sum() for day in previous_days])
                    threshold_value = int(average / target_bars)

                thresholds.append(threshold_value)
        else:
            # If target_bars is None, use the provided threshold
            if threshold is None:
                raise ValueError("Threshold must be provided if target_bars is None.")
            
            thresholds = [threshold] * len(processed_data)
        
        # ======= III. Create the Dataset =======
        resampled_dfs = Parallel(n_jobs=self.n_jobs)(
                delayed(self.get_cumsum_resample)(
                    data=processed_data[idx], 
                    cumsum_column=cumsum_column, 
                    aggregation_rules=aggregation_rules,
                    general_aggregation_rules=general_aggregation_rules, 
                    threshold=thresholds[idx],
                    weighting_column=weighting_column,    
                )
                for idx in range(len(processed_data))
            )

        return resampled_dfs
            
#*____________________________________________________________________________________ #
