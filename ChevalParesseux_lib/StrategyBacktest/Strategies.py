import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

#! ==================================================================================== #
#! ====================================== Base Model ================================== #
class Strategy(ABC):
    def __init__(self):
        # ======= Adapt columns name to the available data / operational consideration =======
        self.date_name = "date"
        self.bid_open_name = "open"
        self.ask_open_name = "open"
        
        # ======= Store Data used for the strategy =======
        self.data = None
        self.processed_data = None
    
    #?____________________________________________________________________________________ #
    def set_names(self, date_name: str, bid_open_name: str, ask_open_name: str):
        """
        Set the names of the columns used in the data, it is important to ensure the operations are done using the correct price. 
        For daily data, it is common to use the open or close price for both bid and ask. The bid-ask spread is usually estimated as a slippage cost.
        
            - date_name (str) : name of the column containing the dates
            - bid_open_name (str) : name of the column containing the bid open prices at which the strategy will operate
            - ask_open_name (str) : name of the column containing the ask open prices at which the strategy will operate
        """
        self.date_name = date_name
        self.bid_open_name = bid_open_name
        self.ask_open_name = ask_open_name
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def set_params(self):
        """This method should be used to set the different parameters of the model."""
        pass
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(self):
        """
        This method should be used to process the data : normalization, feature engineering, etc.
        It is expected to output a list of pd.DataFrame containing the processed data, each element of the list will then be used independtly to predict the signals.
        For daily data, returning a list containing a unique element is enough. Different elements can be used to avoid overnight bias when operating only intraday.
        """
        pass
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def fit(self):
        """
        This method should be used to train the model, optimize hyperparameters and so on.
        It does not explicitly split samples, this has to be done by the user before calling this method.
        """
        pass
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def predict(self):
        """
        This method should be used to predict the signals.
        It is expected to return a pd.DataFrame containing the necessary data to compute the operations (signals, price, date, etc.).
        """
        pass

    #?____________________________________________________________________________________ #
    def operate(self, df: pd.DataFrame):
        """
        This method is common to all strategies and is used to extract the operations from the signals.
        The outputs are the operations (each line corresponds to a different trade) and the signals (each line corresponds to a bar with the associated signal) DataFrames.
        
            - df (pd.DataFrame) : DataFrame containing the data used to extract the operations
        """
        # ======= I. Extract signals =======
        signals_df = self.predict(df=df)
        
        # ======= II. Objects initialization before extracting operations =======
        operations_df = pd.DataFrame(columns=['ID', 'Side', 'Entry_Date', 'Entry_Price', 'Exit_Date', 'Exit_Price', 'PnL'])
        
        # II.1 Set first and last signal to 0 to ensure that the operations are closed
        signals_df.reset_index(drop=True, inplace=True)

        signals_df.loc[0, "signals"] = 0
        signals_df.loc[len(signals_df) - 1, "signals"] = 0
        signals_df.loc[len(signals_df) - 2, "signals"] = 0

        # II.2 Extract the Signal Change and the Entry Points
        signals_df["signal_change"] = signals_df["signals"].diff()
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
            if (current_row["signal_change"] > 0 and previous_row["signals"] == 1):
                side = 1
                entry_date = current_row[self.date_name]
                entry_price = current_row[self.ask_open_name]
                exit_date = next_row[self.date_name]
                exit_price = next_row[self.bid_open_name]
                pnl = (exit_price - entry_price)

            # III.3 Extract Information for a Short Operation
            elif (current_row["signal_change"] < 0 and previous_row["signals"] == -1):
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
