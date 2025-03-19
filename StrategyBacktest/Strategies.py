import sys
sys.path.append("../")
import Data as dt
import Features as ft

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

#! ==================================================================================== #
#! ====================================== Base Model ================================== #
class Strategy(ABC):
    def __init__(self):
        
        self.date_name = "date"
        self.bid_open_name = "open"
        self.ask_open_name = "open"
        
        self.data = None
        self.processed_data = None
    
    #*____________________________________________________________________________________ #
    def set_names(self, date_name: str, bid_open_name: str, ask_open_name: str):
        self.date_name = date_name
        self.bid_open_name = bid_open_name
        self.ask_open_name = ask_open_name
    
    #*____________________________________________________________________________________ #
    @abstractmethod
    def set_params(self):
        pass
    
    #*____________________________________________________________________________________ #
    @abstractmethod
    def load_data(self):
        pass

    #*____________________________________________________________________________________ #
    @abstractmethod
    def process_data(self):
        pass
    
    #*____________________________________________________________________________________ #
    @abstractmethod
    def fit(self):
        pass
    
    #*____________________________________________________________________________________ #
    @abstractmethod
    def predict(self):
        pass

    #*____________________________________________________________________________________ #
    def operate(self, df: pd.DataFrame):
        # ======= I. Extract signals =======
        signals_df = self.predict(df=df)
        
        # ======= II. Objects initialization before extracting operations =======
        operations_df = pd.DataFrame(columns=['ID', 'Side', 'Entry_Date', 'Entry_Price', 'Exit_Date', 'Exit_Price', 'PnL', 'PnL_Percent'])
        
        # II.1 Set first and last signal to 0 to ensure that the operations are closed
        signals_df.reset_index(drop=True, inplace=True)

        signals_df.loc[0, "signal"] = 0
        signals_df.loc[len(signals_df) - 1, "signal"] = 0
        signals_df.loc[len(signals_df) - 2, "signal"] = 0

        # II.2 Extract the Signal Change and the Entry Points
        signals_df["Signal Change"] = signals_df["signal"].diff()
        signals_df["Signal Change"] = signals_df["Signal Change"].shift(1)

        entry_points = signals_df[signals_df["Signal Change"] != 0].copy()
        nb_entry = len(entry_points)
        
        # ======= III. Create an Operation for each entry point =======
        sequential_id = 0
        for idx in range(nb_entry - 1):
            # III.1 Extracting rows
            current_row = entry_points.iloc[idx]
            next_row = entry_points.iloc[idx + 1]
            previous_row = signals_df.iloc[current_row.name - 1]

            # III.2 Extract Information for a Long Operation
            if (current_row["Signal Change"] > 0 and previous_row["signal"] == 1):
                side = 1
                entry_date = current_row[self.date_name]
                entry_price = current_row[self.ask_open_name]
                exit_date = next_row[self.date_name]
                exit_price = next_row[self.bid_open_name]
                pnl = (exit_price - entry_price)
                pnl_percent = pnl / entry_price

            # III.3 Extract Information for a Short Operation
            elif (current_row["Signal Change"] < 0 and previous_row["signal"] == -1):
                side = -1
                entry_date = current_row[self.date_name]
                entry_price = current_row[self.bid_open_name]
                exit_date = next_row[self.date_name]
                exit_price = next_row[self.ask_open_name]
                pnl = (entry_price - exit_price)
                pnl_percent = pnl / entry_price

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
                pnl_percent
            ]

            # --- New sequential id for the next loop iteration ---
            sequential_id += 1
        
        return operations_df, signals_df


#! ==================================================================================== #
#! ====================================== Strategies ================================== #
class MA_crossover(Strategy):
    def __init__(self):
        super().__init__()

        self.window = None
    
    #*____________________________________________________________________________________ #
    def set_params(self, window: int):
        self.window = window
    
    #*____________________________________________________________________________________ #
    def load_data(self, ticker: str, start_date: str, end_date: str):
        data = dt.load_data(ticker)
        data = data.loc[start_date:end_date]
        
        data['date'] = data.index
        data.reset_index(drop=True, inplace=True)

        self.data = data

        return data
    
    #*____________________________________________________________________________________ #
    def process_data(self):
        data = self.data.copy()
        close_series = data['close']

        moving_average = ft.average_features(price_series=close_series, window=self.window)
        data['MA'] = moving_average

        processed_data = [data]

        self.processed_data = processed_data

        return processed_data
    
    #*____________________________________________________________________________________ #
    def fit(self):
        pass

    #*____________________________________________________________________________________ #
    def predict(self, df: pd.DataFrame):
        signals_df = df.copy()
        signals_df['signal'] = 0

        signals_df.loc[signals_df['MA'] < 0, 'signal'] = 1
        signals_df.loc[signals_df['MA'] > 0, 'signal'] = -1

        return signals_df