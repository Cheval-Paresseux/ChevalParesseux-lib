import sys
sys.path.append("../")
from StrategyBacktest import Strategies as st

import pandas as pd
import matplotlib.pyplot as plt

class Backtest():
    def __init__(self, strategy: st.Strategy, flat_cost: float = 0.0, slippage: float = 0.0):
        self.strategy = strategy
        self.data = None
        self.processed_data = None

        self.signals_dfs = None
        self.operations_dfs = None
        self.full_operations_df = None
        self.full_signals_df = None

        self.ticker = None
        self.start_date = None
        self.end_date = None
        self.strategy_params = None
        
        self.flat_cost = flat_cost
        self.slippage = slippage
    
    #*____________________________________________________________________________________ #
    def set_computingParams(self, date_name: str, bid_open_name: str, ask_open_name: str):
        self.strategy.set_names(date_name, bid_open_name, ask_open_name)
    
    #*____________________________________________________________________________________ #
    def set_backtestParams(self, ticker: str, start_date: str, end_date: str, strategy_params: dict):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.strategy_params = strategy_params
    
    #*____________________________________________________________________________________ #
    def run_strategy(self):
        self.strategy.set_params(**self.strategy_params)
        data = self.strategy.load_data(self.ticker, self.start_date, self.end_date)
        processed_data = self.strategy.process_data()
        
        self.data = data
        self.processed_data = processed_data
        
        operations_dfs = []
        signals_dfs = []
        for data_group in processed_data:
            operations_df, signals_df = self.strategy.operate(data_group)
            operations_dfs.append(operations_df)
            signals_dfs.append(signals_df)
        
        full_operations_df = pd.concat(operations_dfs, ignore_index=True, axis=0)
        full_signals_df = pd.concat(signals_dfs, ignore_index=True, axis=0)
        
        return full_operations_df, full_signals_df, operations_dfs, signals_dfs
    
    #*____________________________________________________________________________________ #
    def apply_costs(self, operations_df: pd.DataFrame):
        # ======= I. Ensure there are operations =======
        adjusted_operations_df = operations_df.copy()
        if operations_df.empty:
            return adjusted_operations_df

        # ======= II. Apply slippage on Entry/Exit prices =======
        # II.1 Adjust entry prices
        adjusted_operations_df["Entry_Price_Adjusted"] = adjusted_operations_df.apply(
            lambda row: row["Entry_Price"] * (1 + self.slippage) if row["Side"] == 1 else row["Entry_Price"] * (1 - self.slippage), axis=1
        )

        # II.2 Adjust exit prices
        adjusted_operations_df["Exit_Price_Adjusted"] = adjusted_operations_df.apply(
            lambda row: row["Exit_Price"] * (1 - self.slippage) if row["Side"] == 1 else row["Exit_Price"] * (1 + self.slippage), axis=1
        )

        # ======= III. Adjust the PnL =======
        adjusted_operations_df["PnL_Adjusted"] = (
            adjusted_operations_df["Exit_Price_Adjusted"] - adjusted_operations_df["Entry_Price_Adjusted"]
        ) * adjusted_operations_df["Side"]

        # ======= IV. Apply flat cost =======
        adjusted_operations_df["PnL_Adjusted"] -= self.flat_cost
        adjusted_operations_df["PnL_Percent_Adjusted"] = adjusted_operations_df["PnL_Adjusted"] / adjusted_operations_df["Entry_Price_Adjusted"]

        return adjusted_operations_df

    #*____________________________________________________________________________________ #
    def run_backtest(self):
        full_operations_df, full_signals_df, operations_dfs, signals_dfs = self.run_strategy()
        full_operations_df = self.apply_costs(full_operations_df)
        
        cumulative_returns = (1 + full_operations_df["PnL_Percent"]).cumprod()
        cumulative_returns_adjusted = (1 + full_operations_df["PnL_Percent_Adjusted"]).cumprod()
        
        full_operations_df['Cumulative_PnL'] = cumulative_returns
        full_operations_df['Cumulative_PnL_Adjusted'] = cumulative_returns_adjusted
        
        self.full_operations_df = full_operations_df
        self.full_signals_df = full_signals_df
        self.operations_dfs = operations_dfs
        self.signals_dfs = signals_dfs
        
        return full_operations_df, full_signals_df, operations_dfs, signals_dfs
    
    #*____________________________________________________________________________________ #
    def plot_returns(self):
        
        plt.figure(figsize=(17, 5))
        plt.plot(self.full_operations_df['Cumulative_PnL'], label='Cumulative PnL')
        plt.plot(self.full_operations_df['Cumulative_PnL_Adjusted'], label='Cumulative PnL Adjusted')
        plt.legend()
        plt.show()
        
        