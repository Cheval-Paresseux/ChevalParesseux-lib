
#! ==================================================================================== #
#! =================================== Sampling  ====================================== #
class DataSampler():
    def __init__(
        self, 
        data: pd.DataFrame,
        n_jobs: int = 1
    ):
        # ======= II. Store the inputs =======
        self.data = data
        self.n_jobs = n_jobs
        
        self.available_methods = {
            "daily_volBars": sampl.daily_volBars,
            "daily_cumsumTargetBars": sampl.daily_cumsumTargetBars,
            "daily_cumsumWeightedTargetBars": sampl.daily_cumsumWeightedTargetBars,
        }
        
        # ======= III. Initialize Results =======
        self.params = None
        self.resampled_data = None
    
    #?__________________________________________________________________________________ #
    def set_params(
        self,
        sampling_method: str = "daily_volBars",
        column_name: str = "close",
        grouping_column: str = "date",
        new_cols_methods: str = "mean",
        target_bars: int = 100,
        window_bars_estimation: int = 10,
        pre_threshold: float = 1000,
        weight_column_name: str = "close",
        vol_threshold: float = 0.0005,
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
    ):
        self.params = {
            'sampling_method': sampling_method,
            'column_name': column_name,
            'grouping_column': grouping_column,
            'new_cols_methods': new_cols_methods,
            'target_bars': target_bars,
            'window_bars_estimation': window_bars_estimation,
            'pre_threshold': pre_threshold,
            'weight_column_name': weight_column_name,
            'vol_threshold': vol_threshold,
            'aggregation_dict': aggregation_dict,
        }
        
        return self

    #?____________________________________________________________________________________ #
    def extract(self):
        
        def filter_params_for_function(func, param_dict):
            sig = inspect.signature(func)
            valid_keys = sig.parameters.keys()
            
            return {k: v for k, v in param_dict.items() if k in valid_keys}
        
        # ======= I. Check Sampling Method =======
        if self.params['sampling_method'] not in self.available_methods.keys():
            raise ValueError(f"Sampling method {self.params['sampling_method']} is not available.")
        
        sampling_method = self.available_methods[self.params['sampling_method']]
        
        # ======= II. Apply Sampling =======
        resampled_data = sampling_method(data=self.data, **filter_params_for_function(sampling_method, self.params))
        
        # ======= III. Store Results =======
        self.resampled_data = resampled_data
        
        return resampled_data


#?____________________________________________________________________________________ #
    def vertical_stacking(
        self, 
        dfs_list: list
    ) -> pd.DataFrame:
        """
        Applies vertical stacking to a list of DataFrames.
        
        Parameters:
            - dfs_list (list): List of DataFrames to be stacked.
        
        Returns:
            - stacked_data (pd.DataFrame): The vertically stacked DataFrame.
        """
       # ======= I. Ensure all DataFrames have the same columns =======
        if len(dfs_list) < 1:
            raise ValueError("The list does not contain enough DataFrames.")

        columns = dfs_list[0].columns
        for df in dfs_list[1:]:
            if not df.columns.equals(columns):
                raise ValueError("All DataFrames must have the same columns.")

        # ======= II. Concatenate DataFrames horizontally =======
        stacked_data = pd.concat(dfs_list, axis=0, ignore_index=True)
        self.stacked_data = stacked_data.copy()
        
        return stacked_data