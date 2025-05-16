

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from typing import Union


#! ==================================================================================== #
#! ====================================== Base Model ================================== #
class Strategy(ABC):
    """
    Base class for all strategies.
    
    This class is used to define the common methods and attributes for all strategies.
    Its main purpose is to provide the operate method to extract the operations from the signals.
    """
    #?____________________________________________________________________________________ #
    @abstractmethod
    def __init__(
        self, 
        date_name: str = "date", 
        bid_open_name: str = "bid_open", 
        ask_open_name: str = "ask_open"
    ):
        """
        Constructor of the Strategy class.
        
        Parameters:
            - date_name (str) : Name of the column containing the date
            - bid_open_name (str) : Name of the column containing the bid open price
            - ask_open_name (str) : Name of the column containing the ask open price
        """
        # ======= Adapt columns name to the available data / operational consideration =======
        self.date_name = date_name
        self.bid_open_name = bid_open_name
        self.ask_open_name = ask_open_name
            
    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
        **kwargs
    ) -> Union[tuple, pd.DataFrame, pd.Series]:
        """
        Preprocesses the data before feature extraction.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to be processed.
            - **kwargs: Additional parameters for the data processing.

        Returns:
            - tuple or pd.DataFrame or pd.Series: The processed data ready for feature extraction.
        """
        ...
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def predict(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
        **kwargs
    ) -> pd.DataFrame:
        """
        Core method for signal extraction.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to extract the signals from
            - **kwargs: Additional parameters for the signal extraction.
        
        Returns:
            - pd.DataFrame : The extracted signals as a pd.DataFrame.
        """
        ...

    #?____________________________________________________________________________________ #
    def operate(
        self, 
        data: Union[tuple, pd.Series, pd.DataFrame],
    ):
        """

        """
        # ======= I. Extract signals =======
        try:
            signals_df = self.predict(data=data)
        
        except Exception as e:
            print(f"Error in the predict method: {e}")
            return None, None
        
        # ======= II. Objects initialization before extracting operations =======
        operations_df = pd.DataFrame(columns=['id', 'side', 'size', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'pnl'])
        
        # II.1 Set first and last signal to 0 to ensure that the operations are closed
        signals_df.reset_index(drop=True, inplace=True)

        signals_df.loc[0, "signal"] = 0
        signals_df.loc[len(signals_df) - 1, "signal"] = 0

        # II.2 Extract the Signal Change and the Entry Points
        signals_df["signal_change"] = signals_df["signal"].diff()
        signals_df["signal_change"] = signals_df["signal_change"].shift(1) #! Shifted to avoid look-ahead bias

        entry_points = signals_df[signals_df["signal_change"] != 0].copy()
        nb_entry = len(entry_points)
        
        # ======= III. Create an Operation for each entry point =======
        sequential_id = 0
        for idx in range(nb_entry - 1):
            # III.1 Extracting rows
            current_row = entry_points.iloc[idx]
            next_row = entry_points.iloc[idx + 1]
            previous_row = signals_df.iloc[current_row.name - 1]

            # III.2 Extract Information for a Long Operation
            if (current_row["signal_change"] > 0 and previous_row["signal"] == 1):
                side = 1
                entry_date = current_row[self.date_name]
                entry_price = current_row[self.ask_open_name]
                exit_date = next_row[self.date_name]
                exit_price = next_row[self.bid_open_name]
                pnl = (exit_price - entry_price)

            # III.3 Extract Information for a Short Operation
            elif (current_row["signal_change"] < 0 and previous_row["signal"] == -1):
                side = -1
                entry_date = current_row[self.date_name]
                entry_price = current_row[self.bid_open_name]
                exit_date = next_row[self.date_name]
                exit_price = next_row[self.ask_open_name]
                pnl = (entry_price - exit_price)

            else:
                continue

            # III.4 Append Operation to the DataFrame
            operations_df.loc[sequential_id] = [
                sequential_id,
                side,
                entry_date,
                entry_price,
                exit_date,
                exit_price,
                pnl,
            ]

            # --- New sequential id for the next loop iteration ---
            sequential_id += 1
        
        return operations_df, signals_df

