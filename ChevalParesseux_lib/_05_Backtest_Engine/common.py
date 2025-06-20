
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from typing import Union, Optional, Self
import uuid


#! ==================================================================================== #
#! ================================== Backtest Model ================================== #
class Backtester():
    """
    Backtest Engine for trading strategies.
    
    This class is designed to extract operations from a signals DataFrame and compute the
    corresponding trades, including entry and exit prices, PnL, and other relevant metrics.
    """
    #?_____________________________ Initialization methods _______________________________ #
    def __repr__(self) -> str:
        """
        String representation of the Backtester class.
        
        Returns:
            - str: Description of the Backtester instance.
        """
        return f"Backtester(n_jobs={self.n_jobs}, brokerage_cost={self.brokerage_cost}, slippage_cost={self.slippage_cost})"
    
    #?____________________________________________________________________________________ #
    def __init__(
        self, 
        n_jobs: int = 1,
    ) -> None:
        """
        Constructor for the Backtester class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to run for backtesting. Default is 1.
        """
        # ======= Backtest parameters =======
        self.n_jobs = n_jobs

        # ======= Costs parameters =======
        self.brokerage_cost = None
        self.slippage_cost = None
        
        # ======= Computation parameters =======
        self.ask_name = None
        self.bid_name = None
        self.date_name = None
        
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        brokerage_cost: float = 0.0, 
        slippage_cost: float = 0.0,
        ask_name: str = "ask_open",
        bid_name: str = "bid_open",
        date_name: str = "date",
    ) -> Self:
        """
        Set the parameters for the backtest engine.
        
        Parameters:
            - brokerage_cost (float): Cost of brokerage per trade. Default is 0.0.
            - slippage_cost (float): Slippage cost as a percentage of the trade price. Default is 0.0.
            - ask_name (str): Name of the column containing ask prices. Default is "ask_open".
            - bid_name (str): Name of the column containing bid prices. Default is "bid_open".
            - date_name (str): Name of the column containing dates. Default is "date".
        
        Returns:
            - Self: Returns the instance of the Backtester class with updated parameters.
        """
        # ======= Costs parameters =======
        self.brokerage_cost = brokerage_cost
        self.slippage_cost = slippage_cost

        # ======= Computation parameters =======
        self.ask_name = ask_name
        self.bid_name = bid_name
        self.date_name = date_name

        return self
    
    #?________________________ Operations Extraction methods _____________________________ #
    def add_operation(
        self,
        operations_df: pd.DataFrame,
        code: str,
        trade_id: str,
        side: int,
        size: float,
        size_op: float,
        date: pd.Timestamp,
        entry_price: float,
        exit_price: float,
        unit_pnl: float,
        portfolio_weight: float,
    ) -> pd.DataFrame:
        """
        Auxiliary method to add a new operation to the operations DataFrame.
        
        Parameters:
            - operations_df (pd.DataFrame): The DataFrame containing existing operations.
            - code (str): The code of the asset.
            - trade_id (str): Unique identifier for the trade.
            - side (int): Side of the trade (1 for buy, -1 for sell).
            - size (float): Size of the trade.
            - size_op (float): Size of the operation.
            - date (pd.Timestamp): Date of the operation.
            - entry_price (float): Entry price of the operation.
            - exit_price (float): Exit price of the operation.
            - unit_pnl (float): Profit and Loss of the operation.
            - portfolio_weight (float): Weight of the asset in the portfolio.
        
        Returns:
            - pd.DataFrame: Updated operations DataFrame with the new operation added.
        """
        # ======= I. Create a new row with the operation details =======
        new_row = pd.DataFrame([{
            'code': code,
            'trade_id': trade_id,
            'side': side,
            'size': size,
            'size_op': size_op,
            'date': date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'unit_pnl': unit_pnl,
            'portfolio_weight': portfolio_weight,
        }])
        
        # ======= II. Append the new row to the operations DataFrame =======
        new_operations_df = operations_df.copy()
        new_operations_df.loc[len(new_operations_df)] = new_row.iloc[0]

        return new_operations_df

    #?____________________________________________________________________________________ #
    def get_weighted_entry_price(
        self,
        df: pd.DataFrame, 
        code: str, 
        trade_id: str
    ) -> float:
        """
        Compute the weighted average entry price for a given trade.
        
        Parameters:
            - df (pd.DataFrame): DataFrame containing operations data.
            - code (str): The code of the asset.
            - trade_id (str): Unique identifier for the trade.
        
        Returns:
            - float: Weighted average entry price for the specified trade.
        """
        # ======= I. Extract operations that increased the position =======
        entries = df[(df['code'] == code) & (df['trade_id'] == trade_id) & (df['size_op'] > 0)]
        
        # ======= II. Compute the average price =======
        average_price = np.average(entries['entry_price'], weights=entries['size_op'])
        
        return average_price
    
    #?____________________________________________________________________________________ #
    def get_remaining_position(
        self,
        df: pd.DataFrame, 
        code: str, 
        trade_id: str
    ) -> float:
        """
        Compute the remaining position size for a given trade.
        
        Parameters:
            - df (pd.DataFrame): DataFrame containing operations data.
            - code (str): The code of the asset.
            - trade_id (str): Unique identifier for the trade.
        
        Returns:
            - float: Remaining position size for the specified trade.
        """
        # ======= I. Extract operations for the specified trade and code =======
        entries = df[(df['code'] == code) & (df['trade_id'] == trade_id)]
        
        # ======= II. Compute the remaining size by summing the size_op column =======
        remaining_size = entries['size_op'].sum()

        return remaining_size

    #?____________________________________________________________________________________ #
    def extract_operations(
        self, 
        signals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Main method to extract operations from a signals DataFrame.
        
        Parameters:
            - signals_df (pd.DataFrame): DataFrame containing trading signals with columns:
        
        Returns:
            - pd.DataFrame: DataFrame containing extracted operations with columns:
        """
        # ======= I. Pre-checks =======
        required_columns = [self.bid_name, self.ask_name, self.date_name, 'signal', 'size', 'code', 'portfolio_weight']
        missing = [col for col in required_columns if col not in signals_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # ======= II. Initialize Operations =======
        operations_df = pd.DataFrame({
            'code': pd.Series(dtype='str'),
            'trade_id': pd.Series(dtype='str'),
            'side': pd.Series(dtype='int'),
            'size': pd.Series(dtype='float'),
            'size_op': pd.Series(dtype='float'),
            'date': pd.Series(dtype='datetime64[ns]'),
            'entry_price': pd.Series(dtype='float'),
            'exit_price': pd.Series(dtype='float'),
            'unit_pnl': pd.Series(dtype='float'),
            'portfolio_weight': pd.Series(dtype='float'),
        })

        # ======= II. Ensure signals_df is clean =======
        df = signals_df.copy()
        df['signal'] = df['signal'].fillna(0)
        df['size'] = df['size'].fillna(0)

        df[['signal', 'size']] = df[['signal', 'size']].shift(1) # Avoid leakage
        df = df.iloc[1:].reset_index(drop=True)
        df.loc[df.index[-1], 'signal'] = 0  # Ensure last signal is 0 to avoid dangling operation

        # ======= III. Iterate through the signals to extract operations =======
        code = df['code'].iloc[0]

        last_trade_id = None
        last_side = 0
        last_size = 0  
        last_portfolio_weight = 0.0
        for i, row in df.iterrows():
            # ----- 1. Extract values -----
            side = row['signal']
            size = row['size']
            portfolio_weight = row['portfolio_weight']

            ask_open = row[self.ask_name]
            bid_open = row[self.bid_name]
            date = row[self.date_name]

            # ----- Case 0 : Nothing changed -----
            if (side == last_side) and (size == last_size):
                continue

            # ----- Case 1 : New operation from 0 -----
            elif (side != last_side) and (last_side == 0) and (size != 0):
                # 1. Create the new operation
                trade_id = str(uuid.uuid4())
                size_op = size

                entry_price = ask_open if side == 1 else bid_open
                entry_price = entry_price * (1 + self.slippage_cost) if side == 1 else entry_price * (1 - self.slippage_cost)
                
                exit_price = 0
                unit_pnl = -self.brokerage_cost * size_op * entry_price  # Initial PnL is just the brokerage cost
                
                operations_df = self.add_operation(operations_df, code, trade_id, side, size, size_op, date, entry_price, exit_price, unit_pnl, portfolio_weight)

                # 3. Update variables 
                last_trade_id = trade_id
                last_side = side
                last_size = size
                last_portfolio_weight = portfolio_weight

            # ----- Case 2 : Changing size only -----
            elif (side == last_side) and (size != last_size) and (side != 0):
                # 1. Extract delta size and trade_id
                delta_size = size - last_size
                trade_id = last_trade_id

                # 2. Check if we are adding or reducing the position
                if delta_size > 0: # Adding to position
                    entry_price = ask_open if side == 1 else bid_open
                    entry_price = entry_price * (1 + self.slippage_cost) if side == 1 else entry_price * (1 - self.slippage_cost)
                    
                    exit_price = 0
                    unit_pnl = -self.brokerage_cost * delta_size * entry_price  # Initial PnL is just the brokerage cost

                else: # Reducing position
                    # i. Get the average entry price of this trade
                    average_entry_price = self.get_weighted_entry_price(operations_df, code, last_trade_id)

                    # ii. Create the partial exit operation
                    entry_price = average_entry_price # entry slippage is already included in the average entry price
                    
                    exit_price = bid_open if side == 1 else ask_open
                    exit_price = exit_price * (1 - self.slippage_cost) if side == 1 else exit_price * (1 + self.slippage_cost)
                    
                    unit_pnl = side * (exit_price - average_entry_price) * abs(delta_size) - self.brokerage_cost * abs(delta_size) * exit_price  # PnL is the difference in price times the size, minus brokerage cost
                
                # 3. Create the new operation
                operations_df = self.add_operation(operations_df, code, trade_id, side, size, delta_size, date, entry_price, exit_price, unit_pnl, portfolio_weight)

                # 4. Update variables 
                last_trade_id = trade_id
                last_side = side
                last_size = size

            # ----- Case 3 : New operation in opposite direction -----
            elif (side != last_side) and (last_side != 0):
                # 1. Close the previous operation if it exists
                average_entry_price = self.get_weighted_entry_price(operations_df, code, last_trade_id)
                remaining_size = self.get_remaining_position(operations_df, code, last_trade_id)

                if remaining_size != 0:
                    trade_id = last_trade_id
                    size_op = -remaining_size

                    entry_price = average_entry_price # entry slippage is already included in the average entry price
                    exit_price = bid_open if last_side == 1 else ask_open
                    exit_price = exit_price * (1 - self.slippage_cost) if last_side == 1 else exit_price * (1 + self.slippage_cost)
                    
                    unit_pnl = last_side * (exit_price - average_entry_price) * remaining_size - self.brokerage_cost * remaining_size * exit_price  # PnL is the difference in price times the size, minus brokerage cost

                    operations_df = self.add_operation(operations_df, code, trade_id, last_side, last_size, size_op, date, entry_price, exit_price, unit_pnl, last_portfolio_weight)
                    
                    # 3. Update variables 
                    last_trade_id = trade_id
                    last_side = 0
                    last_size = 0

                # 2. Create the new operation
                if side != 0:
                    trade_id = str(uuid.uuid4())
                    size_op = size
                    
                    entry_price = ask_open if side == 1 else bid_open
                    entry_price = entry_price * (1 + self.slippage_cost) if side == 1 else entry_price * (1 - self.slippage_cost)
                    
                    exit_price = 0
                    unit_pnl = -self.brokerage_cost * size_op * entry_price  # Initial PnL is just the brokerage cost

                    operations_df = self.add_operation(operations_df, code, trade_id, side, size, size_op, date, entry_price, exit_price, unit_pnl, portfolio_weight)

                    # 3. Update variables 
                    last_trade_id = trade_id
                    last_side = side
                    last_size = size
            
        # ======= IV. Finalize operations DataFrame =======
        operations_df['exit_price'] = np.where(operations_df['exit_price'] == 0, np.nan, operations_df['exit_price'])

        return operations_df
    
    #?____________________________ Simulation methods ____________________________________ #
    def get_daily_summary(
        self,
        operations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        """
        # ======= I. Sort operations by date and other relevant columns =======
        aux_df = operations_df.sort_values(by=['date', 'code', 'trade_id']).reset_index(drop=True)

        # ======= II. Split into one DataFrame per day =======
        daily_dfs = [group for _, group in aux_df.groupby('date')]
        
        # ======= III. Extract relevant information from each day's operations =======
        def extract_infos(
            day_df: pd.DataFrame
        ) -> dict:
            """
            Extracts relevant information from a single day's operations DataFrame.
            
            Parameters:
                - day_df (pd.DataFrame): DataFrame containing operations for a single day.
            
            Returns:
                - dict: A dictionary containing the date, number of operations, and daily returns.
            """
            day_infos = {
                'date': day_df['date'].iloc[0],
                'nb_operations': len(day_df),
                'daily_returns': day_df['returns'].sum(),
            }
            return day_infos
        
        daily_infos = Parallel(n_jobs=self.n_jobs)(
            delayed(extract_infos)(day_df) for day_df in daily_dfs
        )
        
        # ======= IV. Create a summary DataFrame with daily information =======
        summary_df = pd.DataFrame(daily_infos)
        summary_df.sort_values(by='date', inplace=True)
        summary_df.set_index('date', inplace=True)
        
        # ======= V. Calculate cumulative returns =======
        summary_df['cumulative_returns'] = (1 + summary_df['daily_returns']).cumprod()

        return summary_df
    
    #?____________________________________________________________________________________ #
    def get_underlying_summary(
        self,
        signals_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        This method extracts the underlying asset's daily closing prices and calculates daily and cumulative returns.
        
        Parameters:
            - signals_df (pd.DataFrame): DataFrame containing trading signals with columns:
                - 'code': Asset code
                - 'date': Date of the signal
                - self.ask_name: Ask price column name (default is "ask_open")
        
        Returns:
            - pd.DataFrame: DataFrame containing daily closing prices, daily returns, and cumulative returns.
        """
        # ======= I. Extract basic information from the signals DataFrame =======
        asset_code = signals_df['code'].iloc[0]
        daily_dfs = [group for _, group in signals_df.groupby('date')]
        
        def extract_close(
            day_df: pd.DataFrame
        ) -> dict:
            """
            Extracts daily returns for a single day's signals DataFrame.
            
            Parameters:
                - day_df (pd.DataFrame): DataFrame containing signals for a single day.
            
            Returns:
                - dict: A dictionary containing the date and closing price for that day.
            """
            close = day_df[self.ask_name].iloc[-1]
            infos = {
                'date': day_df['date'].iloc[0],
                'close': close,
            }
            
            return infos

        # ======= II. Extract daily closing prices =======
        daily_closes = Parallel(n_jobs=self.n_jobs)(
            delayed(extract_close)(day_df) for day_df in daily_dfs
        )
        
        # ======= III. Create a summary DataFrame with daily closing prices =======
        summary_df = pd.DataFrame(daily_closes)
        summary_df.sort_values(by='date', inplace=True)
        summary_df.set_index('date', inplace=True)
        
        # ======= IV. Calculate daily returns and cumulative returns =======
        summary_df['daily_returns'] = summary_df['close'].pct_change().fillna(0)
        summary_df['cumulative_returns'] = (1 + summary_df['daily_returns']).cumprod()
        summary_df['code'] = asset_code
        
        return summary_df
        
    #?______________________________ User methods ________________________________________ #
    def run_backtest(
        self, 
        signals_dfs: Union[list, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Run the backtest on the provided signals DataFrame.
        
        This method extracts operations from the signals DataFrame, computes the corresponding trades, and returns a DataFrame containing the operations along with a daily summary of returns.
        It gives an accurate representation of the trading strategy's performance over time. However, it looks only at the operations and does not account for movements between the opening and the closing 
        of a position. So, it is aimed at analyzing trading strategies that operate a lot and where the price movements between the opening and closing of a position are not significant.
        For instance, it is not suitable for strategies that open positions and hold them for a long time, as it does not account for the price movements during the holding period.
        
        Parameters:
            - signals_dfs (Union[list, pd.DataFrame]): List of DataFrames or a single DataFrame containing trading signals.
        
        Returns:
            - pd.DataFrame: DataFrame containing the extracted operations.
        """
        # ======= I. Extract operations from signals =======
        if isinstance(signals_dfs, pd.DataFrame):
            signals_dfs = [signals_dfs]
        
        operations_dfs = Parallel(n_jobs=self.n_jobs)(
            delayed(self.extract_operations)(signals_df) for signals_df in signals_dfs
        )
        
        # ======= II. Concatenate all operations DataFrames =======
        operations_df = pd.concat(operations_dfs, ignore_index=True)
        operations_df = operations_df.sort_values(by=['date', 'code', 'trade_id']).reset_index(drop=True)
        
        # ======= III. Finalize operations DataFrame =======
        operations_df['returns'] = (operations_df['unit_pnl'] / operations_df['entry_price']) * operations_df['portfolio_weight']
        
        # ======= IV. Extract daily summary =======
        daily_summary_df = self.get_daily_summary(operations_df)
            
        return operations_df, daily_summary_df
    
    #?____________________________________________________________________________________ #
    
        