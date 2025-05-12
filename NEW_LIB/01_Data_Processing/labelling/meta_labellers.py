from ..labelling import common as com

import numpy as np
import pandas as pd
from typing import Union, Self



#! ==================================================================================== #
#! =============================== BINARY LABELLERS ================================== #
class BinaryMeta_labeller(com.Labeller):
    """
    Binary Meta Labeller for discrete time series data.

    It inherits from the Labeller base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_labels : compute the moving average feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "binaryMeta",
        n_jobs: int = 1
    ) -> None:
        """
        Initialize the binaryMeta_labeller.

        Parameters:
            - name (str): Name of the labeller (default is "binaryMeta").
            - n_jobs (int): Number of jobs for parallel processing (default is 1).
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
        )
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        trade_lock: bool = [True, False],
        noZero: bool = [True, False],
    ) -> Self:
        """
        Sets the parameter grid for the labeller.

        Parameters:
            - trade_lock (bool): If True, locks trades to prevent re-entry.
            - noZero (bool): If True, excludes cases where prediction = 0 from being "good".
        """
        self.params = {
            "trade_lock": trade_lock,
            "noZero": noZero,
        }
        
        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: Union[tuple, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Applies preprocessing to the input data before labels extraction.
        
        Parameters:
            - data (tuple | pd.DataFrame): The input data to be processed.
        
        Returns:
            - processed_data (pd.DataFrame): The two series to be used for cointegration testing.
        """
        # ======= I. Extract Series =======
        if isinstance(data, pd.DataFrame):
            nb_series = data.shape[1]
            if nb_series != 2:
                raise ValueError(f"DataFrame must have exactly 2 columns, but got {nb_series}.")
            
            cols = data.columns.to_list()
            if cols[0] == "label" and cols[1] == "signal":
                label_series = data["label"]
                signal_series = data["signal"]
            else:
                label_series = data.iloc[:, 0]
                signal_series = data.iloc[:, 1]
                print("Warning: DataFrame columns are not named 'label' and 'signal'. Using first two columns instead.")
        
        elif isinstance(data, tuple) and len(data) == 2:
            label_series = data[0]
            signal_series = data[1]

        else:
            raise ValueError("Data must be either a tuple of two series or a DataFrame with two columns.")
        
        # ======= II. Ensure Series have the same indexation =======
        processed_data = pd.DataFrame({"label": label_series, "signal": signal_series})
        processed_data = processed_data.dropna()

        return processed_data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        data: Union[tuple, pd.DataFrame],
        trade_lock: bool,
        noZero: bool,
    ) -> pd.Series:
        """
        Compute binary meta labels based on the input data.

        Parameters:
            - data (tuple | pd.DataFrame): The input data to be processed.
            - trade_lock (bool): If True, locks trades to prevent re-entry.
            - noZero (bool): If True, excludes cases where prediction = 0 from being "good".
        
        Returns:
            - label_series (pd.Series): The computed binary labels.
        """
        # ======= I. Process =======
        processed_data = self.process_data(data)

        # ======= II. Compute Meta Labels =======
        if trade_lock and not noZero:
            processed_data["meta_label"] = np.where(processed_data["signal"] == 0, 1, np.where(processed_data["signal"] == processed_data["label"], 1, 0))
            label_series = processed_data["meta_label"]
        
        elif noZero: 
            processed_data["meta_label"] = np.where((processed_data["signal"] == processed_data["label"]) & (processed_data["signal"] != 0), 1, 0)
            label_series = processed_data["meta_label"]
        
        else:
            processed_data["meta_label"] = np.where(processed_data["signal"] == processed_data["label"], 1, 0)
            label_series = processed_data["meta_label"]
        
        # ======= III. Change name =======
        label_series.name = f"{self.name}_{trade_lock}_{noZero}"
        
        return label_series
        


#! ==================================================================================== #
#! =============================== TRINARY LABELLERS =================================== #
class TrinaryMeta_labeller(com.Labeller):
    """
    Trinary Meta Labeller for discrete time series data.

    It inherits from the Labeller base class and implements methods to:
        - set_params : define parameter grids.
        - process_data : optionally performs preprocessing on the input series.
        - get_labels : compute the moving average feature over a rolling window
    """
    def __init__(
        self, 
        name: str = "trinaryMeta",
        n_jobs: int = 1
    ) -> None:
        """
        Initialize the trinaryMeta_labeller.
        
        Parameters:
            - name (str): Name of the labeller (default is "trinaryMeta").
            - n_jobs (int): Number of jobs for parallel processing (default is 1).
        """
        super().__init__(
            name=name,
            n_jobs=n_jobs,
        )
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        extension: bool = [True, False],
        noZero: bool = [True, False],
    ) -> Self:
        """ Set parameters for the labeller.
        
        Parameters:
            - extension (bool): Allows neutral (0,0) to be considered good if False.
            - noZero (bool): Excludes cases where prediction = 0 from being "good".
        """
        self.params = {
            "extension": extension,
            "noZero": noZero,
        }
        
        return self
    
    #?____________________________________________________________________________________ #
    def process_data(
        self, 
        data: Union[tuple, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Applies preprocessing to the input data before labels extraction.
        
        Parameters:
            - data (tuple | pd.DataFrame): The input data to be processed.
        
        Returns:
            - processed_data (pd.DataFrame): The two series to be used for cointegration testing.
        """
        # ======= I. Extract Series =======
        if isinstance(data, pd.DataFrame):
            nb_series = data.shape[1]
            if nb_series != 2:
                raise ValueError(f"DataFrame must have exactly 2 columns, but got {nb_series}.")
            
            cols = data.columns.to_list()
            if cols[0] == "label" and cols[1] == "signal":
                label_series = data["label"]
                signal_series = data["signal"]
            else:
                label_series = data.iloc[:, 0]
                signal_series = data.iloc[:, 1]
                print("Warning: DataFrame columns are not named 'label' and 'signal'. Using first two columns instead.")
        
        elif isinstance(data, tuple) and len(data) == 2:
            label_series = data[0]
            signal_series = data[1]

        else:
            raise ValueError("Data must be either a tuple of two series or a DataFrame with two columns.")
        
        # ======= II. Ensure Series have the same indexation =======
        processed_data = pd.DataFrame({"label": label_series, "signal": signal_series})
        processed_data = processed_data.dropna()

        return processed_data

    #?____________________________________________________________________________________ #
    def get_labels(
        self, 
        data: Union[tuple, pd.DataFrame],
        extension: bool, 
        noZero: bool
    ) -> pd.Series:
        """
        Compute trinary meta labels based on the input data.

        Parameters:
            - data (tuple | pd.DataFrame): The input data to be processed.
            - trade_lock (bool): If True, locks trades to prevent re-entry.
            - noZero (bool): If True, excludes cases where prediction = 0 from being "good".
        
        Returns:
            - label_series (pd.Series): The computed trinary labels.
        """
        # ======= I. Process =======
        processed_data = self.process_data(data)

        # ======= II. Compute Meta Labels =======
        is_good = (processed_data["signal"] == processed_data["label"])
        is_neutral = (processed_data["signal"] == 0) & (processed_data["label"] != 0)
        is_ugly = ((processed_data["signal"] == -1) & (processed_data["label"] == 1)) | ((processed_data["signal"] == 1) & (processed_data["label"] == -1))
        
        # ======= II. Adjustments =======
        if noZero:
            is_good &= (processed_data["signal"] != 0)

        if extension:
            is_good |= (processed_data["signal"] == 0) & (processed_data["label"] == 0)

        # ======= III. Labelling =======
        processed_data["meta_label"] = np.select([is_good, is_neutral, is_ugly], [1, 0, -1], default=0)
        labels_series = processed_data["meta_label"]
        
        # ======= IV. Change name =======
        labels_series.name = f"{self.name}_{extension}_{noZero}"
        
        return labels_series
        
#*____________________________________________________________________________________ #

