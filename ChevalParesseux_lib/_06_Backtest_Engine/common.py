
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from typing import Union, Optional, Self
import uuid


#! ==================================================================================== #
#! ================================== Backtest Model ================================== #
class Backtester():
    #?_____________________________ Initialization methods _______________________________ #
    def __init__(
        self, 
        n_jobs: int = 1,
    ) -> None:
        """
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
        pnl: float,
    ) -> pd.DataFrame:

        # --- Define the new row ---
        new_row = pd.DataFrame([{
            'code': code,
            'trade_id': trade_id,
            'side': side,
            'size': size,
            'size_op': size_op,
            'date': date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
        }])
        # --- Append the new row to the operations DataFrame ---
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
        
        entries = df[(df['code'] == code) & (df['trade_id'] == trade_id) & (df['size_op'] > 0)]
        average_price = np.average(entries['entry_price'], weights=entries['size_op'])
        
        return average_price
    
    #?____________________________________________________________________________________ #
    def get_remaining_position(
        self,
        df: pd.DataFrame, 
        code: str, 
        trade_id: str
    ) -> float:
        
        entries = df[(df['code'] == code) & (df['trade_id'] == trade_id)]
        remaining_size = entries['size_op'].sum()

        return remaining_size

    #?____________________________________________________________________________________ #
    def extract_operations(
        self, 
        signals_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        """
        # ======= I. Pre-checks =======
        required_columns = [self.bid_name, self.ask_name, self.date_name, 'signal', 'size', 'code']
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
            'pnl': pd.Series(dtype='float'),
        })

        # ======= II. Ensure signals_df is clean =======
        df = signals_df.copy()
        df['signal'] = df['signal'].fillna(0)
        df['size'] = df['size'].fillna(0)

        df[['signal', 'size']] = signals_df[['signal', 'size']].shift(1) # Avoid leakage
        df = df.iloc[1:].reset_index(drop=True)
        df.loc[df.index[-1], 'signal'] = 0  # Ensure last signal is 0 to avoid dangling operation

        # ======= III. Iterate through the signals to extract operations =======
        code = df['code'].iloc[0]

        last_trade_id = None
        last_side = 0
        last_size = 0  
        for i, row in df.iterrows():
            # ----- 1. Extract values -----
            side = row['signal']
            size = row['size']

            ask_open = row[self.ask_name]
            bid_open = row[self.bid_name]
            date = row[self.date_name]

            # ----- Case 0 : Nothing changed -----
            if (side == last_side) and (size == last_size):
                continue

            # ----- Case 1 : New operation from 0 -----
            elif (side != last_side) and (last_side == 0):
                # 1. Create the new operation
                trade_id = str(uuid.uuid4())
                size_op = size

                entry_price = ask_open if side == 1 else bid_open
                entry_price = entry_price * (1 + self.slippage_cost) if side == 1 else entry_price * (1 - self.slippage_cost)
                
                exit_price = 0
                pnl = -self.brokerage_cost * size_op * entry_price  # Initial PnL is just the brokerage cost
                
                operations_df = self.add_operation(operations_df, code, trade_id, side, size, size_op, date, entry_price, exit_price, pnl)

                # 3. Update variables 
                last_trade_id = trade_id
                last_side = side
                last_size = size

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
                    pnl = -self.brokerage_cost * delta_size * entry_price  # Initial PnL is just the brokerage cost

                else: # Reducing position
                    # i. Get the average entry price of this trade
                    average_entry_price = self.get_weighted_entry_price(operations_df, code, last_trade_id)

                    # ii. Create the partial exit operation
                    entry_price = average_entry_price # entry slippage is already included in the average entry price
                    
                    exit_price = bid_open if side == 1 else ask_open
                    exit_price = exit_price * (1 - self.slippage_cost) if side == 1 else exit_price * (1 + self.slippage_cost)
                    
                    pnl = side * (average_entry_price - (exit_price)) * abs(delta_size) - self.brokerage_cost * abs(delta_size) * entry_price  # PnL is the difference in price times the size, minus brokerage cost
                
                # 3. Create the new operation
                operations_df = self.add_operation(operations_df, code, trade_id, side, size, delta_size, date, entry_price, exit_price, pnl)

                # 4. Update variables 
                last_trade_id = trade_id
                last_side = side
                last_size = size

            # ----- Case 3 : New operation in opposite direction -----
            elif (side != last_side) and (last_side != 0):
                # 1. Close the previous operation if it exists
                average_entry_price = self.get_weighted_entry_price(operations_df, code, last_trade_id)
                remaining_size = -self.get_remaining_position(operations_df, code, last_trade_id)

                if remaining_size != 0:
                    trade_id = last_trade_id
                    size_op = remaining_size

                    entry_price = average_entry_price # entry slippage is already included in the average entry price
                    exit_price = bid_open if last_side == 1 else ask_open
                    exit_price = exit_price * (1 - self.slippage_cost) if last_side == 1 else exit_price * (1 + self.slippage_cost)
                    
                    pnl = last_side * (average_entry_price - exit_price) * remaining_size - self.brokerage_cost * abs(remaining_size) * entry_price  # PnL is the difference in price times the size, minus brokerage cost

                    operations_df = self.add_operation(operations_df, code, trade_id, last_side, last_size, size_op, date, entry_price, exit_price, pnl)

                # 2. Create the new operation
                if side != 0:
                    trade_id = str(uuid.uuid4())
                    size_op = size
                    
                    entry_price = ask_open if side == 1 else bid_open
                    entry_price = entry_price * (1 + self.slippage_cost) if side == 1 else entry_price * (1 - self.slippage_cost)
                    
                    exit_price = 0
                    pnl = -self.brokerage_cost * size_op * entry_price  # Initial PnL is just the brokerage cost

                    operations_df = self.add_operation(operations_df, code, trade_id, side, size, size_op, date, entry_price, exit_price, pnl)

                    # 3. Update variables 
                    last_trade_id = trade_id
                    last_side = side
                    last_size = size
            
        # ======= IV. Finalize operations DataFrame =======
        operations_df['exit_price'] = np.where(operations_df['exit_price'] == 0, np.nan, operations_df['exit_price'])

        return operations_df
    
    #?______________________________ User methods ________________________________________ #
    