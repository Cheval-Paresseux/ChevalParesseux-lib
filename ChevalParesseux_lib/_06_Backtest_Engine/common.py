
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from typing import Union, Optional, Self



#! ==================================================================================== #
#! ================================== Backtest Model ================================== #
class Backtester():
    #?_____________________________ Initialization methods _______________________________ #
    def __init__(
        self, 
        strategy: object,
        initial_capital: float = 1.0,
    ) -> None:
        """
        Constructor for the Backtester class.
        
        Parameters:
            - strategy (object): The strategy object to be used for backtesting.
            - initial_capital (float): The initial capital for the backtest.
        """
        # ======= Backtest parameters =======
        self.brokerage_cost = None
        self.slippage_cost = None
        
        self.n_jobs = 1
        
        # ======= Strategy inputs =======
        self.strategy = strategy
        self.initial_capital = initial_capital
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        brokerage_cost: float = 0.0, 
        slippage_cost: float = 0.0,
        n_jobs: int = 1
    ) -> Self:
        """
        Sets the parameters for the backtest.
        
        Parameters:
            - brokerage_cost (float): The brokerage cost per trade.
            - slippage_cost (float): The slippage cost per trade.
            - n_jobs (int): The number of jobs to run in parallel.
        """
        self.n_jobs = n_jobs
        self.brokerage_cost = brokerage_cost
        self.slippage_cost = slippage_cost

        return self
    
    #?________________________________ Auxiliary methods _________________________________ #
    def run_strategy(
        self,
        data: Union[pd.DataFrame, list]
    ) -> tuple:
        """
        This method runs the strategy on the provided data.
        
        Parameters:
            - data (Union[pd.DataFrame, list]): The data to be used for backtesting. It can be a single DataFrame or a list of DataFrames.
        
        Returns:
            - tuple: A tuple containing the full operations DataFrame, full signals DataFrame, individual operations DataFrames, and individual signals DataFrames.
        """
        # ======= I. Ensure Data is in the right format =======
        if isinstance(data, pd.DataFrame):
            prepared_data = [data]
        elif isinstance(data, list):
            prepared_data = data
        else:
            raise ValueError("Data must be a pandas DataFrame or a list of DataFrames.")
        
        # ======= II. Run the Strategy =======
        if self.n_jobs > 1:
            #! Be aware that the strategy should be thread-safe and to keep track of the timestamps to reconstitute the operations later.
            operations_dfs, signals_dfs = Parallel(n_jobs=self.n_jobs)(delayed(self.strategy.operate)(data_group) for data_group in prepared_data)
        else:
            operations_dfs = []
            signals_dfs = []
            for data_group in prepared_data:
                operations_df, signals_df = self.strategy.operate(data_group)
                operations_dfs.append(operations_df)
                signals_dfs.append(signals_df)
        
        return operations_dfs, signals_dfs
    
    #?____________________________________________________________________________________ #
    def apply_costs(
        self, 
        operations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        This method applies slippage to the entry and exit prices of the operations.
        
        Parameters:
            - operations_df (pd.DataFrame): The DataFrame containing the operations data.
        
        Returns:
            - pd.DataFrame: The adjusted operations DataFrame with slippage applied.
        """
        # ======= I. Ensure there are operations =======
        adjusted_df = operations_df.copy()
        if operations_df.empty:
            return adjusted_df

        # ======= II. Apply slippage on Entry/Exit prices =======
        # ----- 1. Initialize variables -----
        cumulative_entry_value = 0
        cumulative_exit_value = 0
        cumulative_size = 0
        cumulative_brokerage = 0
        last_trade_id = None
        last_size = 0

        # ----- 2. Iterate through each operation -----
        for idx, row in adjusted_df.iterrows():
            # 2.1 Extract trade information
            trade_id = row["trade_id"]
            side = row["side"]
            entry_price = row["entry_price"]
            exit_price = row["exit_price"]
            size = row["size"]

            # 2.2 Reset cumulative values if trade_id changes (=> new trade)
            if trade_id != last_trade_id:
                cumulative_entry_value = 0
                cumulative_exit_value = 0
                cumulative_size = 0
                cumulative_brokerage = 0
                size_change = size
            else:
                size_change = size - last_size
                
            # 2.3 Calculate the absolute change in size
            abs_change = np.abs(size_change)
            if abs_change > 0:
                # 2.4 Adjust entry and exit prices based on slippage
                if side == 1:
                    entry_adj = entry_price * (1 + self.slippage_cost)
                    exit_adj = exit_price * (1 - self.slippage_cost)
                else:
                    entry_adj = entry_price * (1 - self.slippage_cost)
                    exit_adj = exit_price * (1 + self.slippage_cost)

                cumulative_entry_value += entry_adj * abs_change
                cumulative_exit_value += exit_adj * abs_change
                cumulative_size += abs_change

                # 2.5 Calculate the brokerage cost
                cumulative_brokerage += self.brokerage_cost * abs_change * entry_price

                # 2.6 Adjust the entry and exit prices
                entry_price_adjusted = cumulative_entry_value / cumulative_size
                exit_price_adjusted = cumulative_exit_value / cumulative_size
                pnl_adjusted = ((cumulative_exit_value - cumulative_entry_value) / cumulative_size * side) - (cumulative_brokerage / cumulative_size)

                adjusted_df.at[idx, "entry_price_adjusted"] = entry_price_adjusted
                adjusted_df.at[idx, "exit_price_adjusted"] = exit_price_adjusted
                adjusted_df.at[idx, "pnl_adjusted"] = pnl_adjusted
            
            # 2.7 Actualize the last trade information
            last_trade_id = trade_id
            last_size = size

        return adjusted_df
    
    #?_____________________________ User Functions _______________________________________ #
    def run_backtest(
        self, 
        data: pd.DataFrame
    ):
        # ======= I. Run the Strategy =======
        operations_dfs, signals_dfs = self.run_strategy(data=data)
        
        # ======= II. Apply slippage and brokerage costs =======
        for i, operations_df in enumerate(operations_dfs):
            operations_dfs[i] = self.apply_costs(operations_df)
        
        # ======= III. Concatenate the DataFrames =======
        full_operations_df = pd.concat(operations_dfs, ignore_index=True)
        full_operations_df = full_operations_df.sort_values(by=['entry_date', 'trade_id']).reset_index(drop=True)
        
        # ======= IV. Compute the Daily and Cumulative Returns (No intra-day compounding) =======
        aux_df = full_operations_df.copy()
        aux_df['returns'] = aux_df['pnl_adjusted'] / aux_df['entry_price_adjusted']
        
        # Daily net PnL (no intra-day compounding)
        aux_df['entry_date'] = pd.to_datetime(aux_df['entry_date'])
        daily_operations = aux_df.groupby('entry_date').to_list()

        for i, daily_operations_df in enumerate(daily_operations):
            continue 
        #TODO: Implement the logic to compute daily operations and returns

        return full_operations_df, signals_dfs, returns_df
        
    #?____________________________________________________________________________________ #
    def plot_operationsBars(
        self, 
        by_date: bool = False, 
        buyHold: bool = True, 
        noFees: bool = True, 
        fees: bool = True
    ):
        # ======= I. Prepare the DataFrame for plotting =======
        plotting_df = self.full_operations_df.copy()

        # ======= II. Initialize the plot =======
        sns.set_style("whitegrid")
        colors = sns.color_palette("husl", 3)
        plt.figure(figsize=(17, 6))
        
        if by_date:
            plotting_df = plotting_df.set_index(plotting_df['entry_date'])
            plt.xlabel('Date', fontsize=14, fontweight='bold')
        else:
            plt.xlabel('Number of Trades', fontsize=14, fontweight='bold')
        
        plt.ylabel('Cumulative Returns', fontsize=14, fontweight='bold')
        plt.title('Strategy Performance Comparison', fontsize=16, fontweight='bold')

        # ======= III. Plot the Cumulative Returns =======
        if buyHold:
            plt.plot(plotting_df['buyHold_cumret'], label='Buy and Hold', color=colors[0], linewidth=2)
        if noFees:
            plt.plot(plotting_df['noFees_strategy_cumret'], label='Cumulative Returns Without Fees', color=colors[1], linestyle='--', linewidth=1)
        if fees:
            plt.plot(plotting_df['strategy_cumret'], label='Cumulative Returns Adjusted', color=colors[2], linewidth=2)

        plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        # ======= IV. Compute statistics =======
        returns_series = plotting_df['strategy_returns']
        market_returns = plotting_df['buyHold_returns']

        performance_stats, _ = ft.get_performance_measures(returns_series, market_returns, frequence="daily")

        return performance_stats
    
    #?____________________________________________________________________________________ #
    def plot_timeBars(
        self
    ):
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
        plt.plot(plotting_df['buyHold_cumret'], label='Buy and Hold', color=colors[0], linewidth=2)
        plt.plot(plotting_df['strategy_cumret'], label='Cumulative Returns Adjusted', color=colors[2], linewidth=2)

        plt.legend(fontsize=12, loc='best', frameon=True, shadow=True)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        # ======= IV. Compute statistics =======
        returns_series = plotting_df['strategy_returns']
        market_returns = plotting_df['buyHold_returns']

        performance_stats, _ = ft.get_performance_measures(returns_series, market_returns, frequence="daily")

        return performance_stats
        