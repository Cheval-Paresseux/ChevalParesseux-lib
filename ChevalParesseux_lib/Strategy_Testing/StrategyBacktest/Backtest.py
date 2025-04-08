from ..StrategyBacktest import Strategies as st
from ..utils import financialMetrics as ft

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed

#! ==================================================================================== #
#! ================================== Backtest Model ================================== #
class Backtest():
    def __init__(self, strategy: st.Strategy):
        # ======= Backtest parameters =======
        self.brokerage_cost = None
        self.slippage_cost = None
        
        self.n_jobs = 1
        
        # ======= Strategy inputs=======
        self.strategy = strategy
        self.strategy_params = None
        self.data = None
        self.processed_data = None

        # ======= Backtest results =======
        self.signals_dfs = None
        self.operations_dfs = None
        self.full_operations_df = None
        self.full_signals_df = None

    #?_____________________________ Params Functions _____________________________________ #
    def set_computingParams(self, date_name: str, bid_open_name: str, ask_open_name: str, n_jobs: int = 1):
        """
        This method is used to set the different parameters to ensure the correct computation of the operations.
        
            - date_name (str) : name of the column containing the dates
            - bid_open_name (str) : name of the column containing the bid open prices at which the strategy will operate
            - ask_open_name (str) : name of the column containing the ask open prices at which the strategy will operate
            - n_jobs (int) : number of jobs to run in parallel
        """
        self.strategy.set_names(date_name=date_name, bid_open_name=bid_open_name, ask_open_name=ask_open_name)
        self.n_jobs = n_jobs
    
    #?____________________________________________________________________________________ #
    def set_backtestParams(self, strategy_params: dict, brokerage_cost: float = 0.0, slippage_cost: float = 0.0):
        
        self.strategy_params = strategy_params
        self.brokerage_cost = brokerage_cost
        self.slippage_cost = slippage_cost
    
    #?_____________________________ Build Functions ______________________________________ #
    def run_strategy(self):
        # ======= I. Set up Parameters and Data =======
        data = self.data.copy()
        self.strategy.set_params(**self.strategy_params)
        processed_data = self.strategy.process_data(data)
        
        # I.2 Store the data
        self.processed_data = processed_data
        
        # ======= II. Run the Strategy =======
        if self.n_jobs > 1:
            #! Be aware that the strategy should be thread-safe and to keep track of the timestamps to reconstitute the operations later.
            operations_dfs, signals_dfs = Parallel(n_jobs=self.n_jobs)(delayed(self.strategy.operate)(data_group) for data_group in processed_data)
        else:
            operations_dfs = []
            signals_dfs = []
            for data_group in processed_data:
                operations_df, signals_df = self.strategy.operate(data_group)
                operations_dfs.append(operations_df)
                signals_dfs.append(signals_df)
        
        full_operations_df = pd.concat(operations_dfs, ignore_index=True, axis=0)
        full_signals_df = pd.concat(signals_dfs, ignore_index=True, axis=0)
        
        return full_operations_df, full_signals_df, operations_dfs, signals_dfs
    
    #?____________________________________________________________________________________ #
    def apply_slippage(self, operations_df: pd.DataFrame):
        """
        This method is used to apply the slippage on the operations by modifying the entry and exit prices.
        
            - operations_df (pd.DataFrame) : DataFrame containing the operations
        """
        # ======= I. Ensure there are operations =======
        adjusted_operations_df = operations_df.copy()
        if operations_df.empty:
            return adjusted_operations_df

        # ======= II. Apply slippage on Entry/Exit prices =======
        # II.1 Adjust entry prices
        adjusted_operations_df["Entry_Price_Adjusted"] = adjusted_operations_df.apply(
            lambda row: row["Entry_Price"] * (1 + self.slippage_cost) if row["Side"] == 1 else row["Entry_Price"] * (1 - self.slippage_cost), axis=1
        )

        # II.2 Adjust exit prices
        adjusted_operations_df["Exit_Price_Adjusted"] = adjusted_operations_df.apply(
            lambda row: row["Exit_Price"] * (1 - self.slippage_cost) if row["Side"] == 1 else row["Exit_Price"] * (1 + self.slippage_cost), axis=1
        )

        # ======= III. Adjust the PnL =======
        adjusted_operations_df["PnL_Adjusted"] = (adjusted_operations_df["Exit_Price_Adjusted"] - adjusted_operations_df["Entry_Price_Adjusted"]) * adjusted_operations_df["Side"]

        return adjusted_operations_df
    
    #?____________________________________________________________________________________ #
    def apply_brokerage(self, operations_df: pd.DataFrame):
        """
        This method is used to apply the brokerage cost on the operations by modifying the PnL.
        
            - operations_df (pd.DataFrame) : DataFrame containing the operations
        """
        # ======= I. Ensure there are operations =======
        adjusted_operations_df = operations_df.copy()
        if operations_df.empty:
            return adjusted_operations_df

        # ======= II. Apply brokerage on PnL =======
        adjusted_operations_df["PnL_Adjusted"] = adjusted_operations_df["PnL_Adjusted"] - (self.brokerage_cost * np.abs(adjusted_operations_df["Entry_Price"]))

        return adjusted_operations_df

    #?_____________________________ User Functions _______________________________________ #
    def run_backtest(self, data: pd.DataFrame):
        """
        This method is used to run the backtest.
        """
        # ======= 0. Initialize Data =======
        self.data = data
        
        # ======= I. Run the Strategy =======
        full_operations_df, full_signals_df, operations_dfs, signals_dfs = self.run_strategy()
        full_operations_df = self.apply_slippage(full_operations_df)
        full_operations_df = self.apply_brokerage(full_operations_df)
        
        # ======= II. Compute the Cumulative Returns : Operations bars =======
        # II.1 Without Fees
        full_operations_df['NoFees_strategy_returns'] = full_operations_df['PnL'] / full_operations_df['Entry_Price']
        full_operations_df['NoFees_strategy_cumret'] = (1 + full_operations_df['NoFees_strategy_returns']).cumprod()
        
        # II.2 With Fees
        full_operations_df['strategy_returns'] = full_operations_df['PnL_Adjusted'] / full_operations_df['Entry_Price']
        full_operations_df['strategy_cumret'] = (1 + full_operations_df['strategy_returns']).cumprod()
        
        # II.3 Buy and Hold
        full_operations_df['BuyAndHold_returns'] = (full_operations_df['Exit_Price'] - full_operations_df['Entry_Price']) / full_operations_df['Entry_Price']
        full_operations_df['BuyAndHold_cumret'] = (1 + full_operations_df['BuyAndHold_returns']).cumprod()

        # ======= III. Compute the Cumulative Returns : Time bars =======
        # For this part, we don't consider the fees and slippage as this computation is relevant only for very low frequency strategies which are less impacted by these costs.
        name_series = self.strategy.ask_open_name
        full_signals_df['BuyAndHold_returns'] = (full_signals_df[name_series].shift(-1) - full_signals_df[name_series]) / full_signals_df[name_series]
        full_signals_df['BuyAndHold_cumret'] = (1 + full_signals_df['BuyAndHold_returns']).cumprod()
        full_signals_df['strategy_returns'] = full_signals_df['signals'] * full_signals_df['BuyAndHold_returns'].shift(-1)
        full_signals_df['strategy_cumret'] = (1 + full_signals_df['strategy_returns']).cumprod()
        
        # ======= IV. Store the results =======
        self.full_operations_df = full_operations_df
        self.full_signals_df = full_signals_df
        self.operations_dfs = operations_dfs
        self.signals_dfs = signals_dfs
        
        return full_operations_df, full_signals_df
        
    #?____________________________________________________________________________________ #
    def plot_operationsBars(self, by_date: bool = False, BuyAndHold: bool = True, NoFees: bool = True, Fees: bool = True):
        """
        Plots the strategy's cumulative returns based on executed trades.  
        This method intentionally excludes daily portfolio valuation to avoid overestimating result significance.
        """
        # ======= I. Prepare the DataFrame for plotting =======
        plotting_df = self.full_operations_df.copy()

        # ======= II. Initialize the plot =======
        sns.set_style("whitegrid")
        colors = sns.color_palette("husl", 3)
        plt.figure(figsize=(17, 6))
        
        if by_date:
            plotting_df = plotting_df.set_index(plotting_df['Entry_Date'])
            plt.xlabel('Date', fontsize=14, fontweight='bold')
        else:
            plt.xlabel('Number of Trades', fontsize=14, fontweight='bold')
        
        plt.ylabel('Cumulative Returns', fontsize=14, fontweight='bold')
        plt.title('Strategy Performance Comparison', fontsize=16, fontweight='bold')

        # ======= III. Plot the Cumulative Returns =======
        if BuyAndHold:
            plt.plot(plotting_df['BuyAndHold_cumret'], label='Buy and Hold', color=colors[0], linewidth=2)
        if NoFees:
            plt.plot(plotting_df['NoFees_strategy_cumret'], label='Cumulative Returns Without Fees', color=colors[1], linestyle='--', linewidth=1)
        if Fees:
            plt.plot(plotting_df['strategy_cumret'], label='Cumulative Returns Adjusted', color=colors[2], linewidth=2)

        plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        # ======= IV. Compute statistics =======
        returns_series = plotting_df['strategy_returns']
        market_returns = plotting_df['BuyAndHold_returns']

        performance_stats, _ = ft.get_performance_measures(returns_series, market_returns, frequence="daily")

        return performance_stats
    
    #?____________________________________________________________________________________ #
    def plot_timeBars(self):
        """
        This method is used to plot the strategy's cumulative returns based on time bars.
        """
        # ======= I. Prepare the DataFrame for plotting =======
        date_name = self.strategy.date_name
        plotting_df = self.full_signals_df.copy()
        plotting_df = plotting_df.set_index(plotting_df[date_name])

        # ======= II. Initialize the plot =======
        sns.set_style("whitegrid")
        colors = sns.color_palette("husl", 3)
        plt.figure(figsize=(17, 6))
        
        plt.xlabel('Date', fontsize=14, fontweight='bold')
        plt.ylabel('Cumulative Returns', fontsize=14, fontweight='bold')
        plt.title('Strategy Performance Comparison', fontsize=16, fontweight='bold')

        # ======= III. Plot the Cumulative Returns =======
        plt.plot(plotting_df['BuyAndHold_cumret'], label='Buy and Hold', color=colors[0], linewidth=2)
        plt.plot(plotting_df['strategy_cumret'], label='Cumulative Returns Adjusted', color=colors[2], linewidth=2)

        plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        # ======= IV. Compute statistics =======
        returns_series = plotting_df['strategy_returns']
        market_returns = plotting_df['BuyAndHold_returns']

        performance_stats, _ = ft.get_performance_measures(returns_series, market_returns, frequence="daily")

        return performance_stats
        