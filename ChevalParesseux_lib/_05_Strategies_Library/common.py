import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from typing import Union, Self
import uuid



#! ==================================================================================== #
#! ====================================== Base Model ================================== #
class Strategy(ABC):
    """
    Base class for all strategies.
    
    This class is used to define the common methods and attributes for all strategies.
    Its main purpose is to provide the operate method to extract the operations from the signals.
    """
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int = 1,
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
        # ======= I. Jobs =======
        self.n_jobs = n_jobs

        # ======= II. Adapt columns name to the available data  =======
        self.date_name = date_name
        self.bid_open_name = bid_open_name
        self.ask_open_name = ask_open_name

    #?____________________________________________________________________________________ #
    @abstractmethod
    def set_params(
        self,
        **kwargs
    ) -> Self:
        """
        Sets the parameter of the strategy.

        Parameters:
            - **kwargs: additional parameters for the strategy.

        Returns:
            - Self: The instance of the class with the parameter grid set.
        """
        ...  

    #?________________________________ Auxiliary methods _________________________________ #
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
    ) -> Union[list, pd.DataFrame]:
        """
        Core method for signal extraction.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to extract the signals from
            - **kwargs: Additional parameters for the signal extraction.
        
        Returns:
            - list or pd.DataFrame: The extracted signals.
        """
        ...

    #?__________________________________ Build methods ___________________________________ #
    def create_operations(
        self,
        signals_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate the operations DataFrame from a signals DataFrame.

        Parameters:
            - signals_df (pd.DataFrame): DataFrame containing signals and sizes.

        Returns:
            - operations_df (pd.DataFrame): DataFrame containing the operations.
        """

        # ======= I. Create the operations DataFrame =======
        operations_df = pd.DataFrame(columns=[
            'trade_id', 'side', 'size',
            'entry_date', 'entry_price',
            'exit_date', 'exit_price',
            'pnl'
        ])
        
        # ======= II. Ensure signals_df is right =======
        signals_df = signals_df.copy()
        signals_df['signal'] = signals_df['signal'].fillna(0)
        signals_df['size'] = signals_df['size'].fillna(0)

        signals_df[['signal', 'size']] = signals_df[['signal', 'size']].shift(1)
        signals_df = signals_df.iloc[1:].reset_index(drop=True)

        # ======= III. Create the operations =======
        position_open = False
        current_side = 0
        current_size = 0
        entry_date = None
        entry_price = None

        sequential_idx = 0
        trade_id = str(uuid.uuid4())

        for i in range(len(signals_df)):
            # ----- 1. Get the row informations-----
            row = signals_df.iloc[i]
            date = row[self.date_name]
            signal = row["signal"]
            size = row["size"]

            ask_open = row[self.ask_open_name]
            bid_open = row[self.bid_open_name]

            # ----- 2. Check if we need to open a position -----
            if not position_open and signal != 0:
                position_open = True
                current_side = signal
                current_size = size
                entry_date = date
                entry_price = ask_open if signal == 1 else bid_open
                continue

            # ----- 3. Check if we need to close a position -----
            if position_open:
                # Case 1 : signal becomes 0 → closing position
                if signal == 0:
                    exit_date = date
                    exit_price = bid_open if current_side == 1 else ask_open
                    pnl = current_size * (exit_price - entry_price) if current_side == 1 else current_size * (entry_price - exit_price)

                    operations_df.loc[sequential_idx] = [
                        trade_id, current_side, current_size,
                        entry_date, entry_price,
                        exit_date, exit_price,
                        pnl
                    ]
                    sequential_idx += 1
                    trade_id = str(uuid.uuid4())
                    position_open = False
                    current_side = 0
                    continue

                # Case 2 : directional change → closing position and opening new one
                if signal != current_side:
                    # 1. Closing
                    exit_date = date
                    exit_price = bid_open if current_side == 1 else ask_open
                    pnl = current_size * (exit_price - entry_price) if current_side == 1 else current_size * (entry_price - exit_price)

                    operations_df.loc[sequential_idx] = [
                        trade_id, current_side, current_size,
                        entry_date, entry_price,
                        exit_date, exit_price,
                        pnl
                    ]
                    sequential_idx += 1
                    trade_id = str(uuid.uuid4())

                    # 2. Opening new position
                    current_side = signal
                    current_size = size
                    entry_date = date
                    entry_price = ask_open if signal == 1 else bid_open
                    continue

                # Case 3 : same direction but different size
                if size != current_size:
                    # 1. Full closing + reopening
                    exit_date = date
                    exit_price = bid_open if current_side == 1 else ask_open
                    pnl = current_size * (exit_price - entry_price) if current_side == 1 else current_size * (entry_price - exit_price)

                    operations_df.loc[sequential_idx] = [
                        trade_id, current_side, current_size,
                        entry_date, entry_price,
                        exit_date, exit_price,
                        pnl
                    ]
                    sequential_idx += 1

                    # Immediately opening new position with new size
                    current_size = size
                    entry_date = date
                    entry_price = ask_open if current_side == 1 else bid_open

        # Security check: if position is still open at the end of the loop, close it
        if position_open:
            last_row = signals_df.iloc[-1]
            exit_date = last_row[self.date_name]
            exit_price = last_row[self.bid_open_name] if current_side == 1 else last_row[self.ask_open_name]
            pnl = current_size * (exit_price - entry_price) if current_side == 1 else current_size * (entry_price - exit_price)

            operations_df.loc[sequential_idx] = [
                trade_id, current_side, current_size,
                entry_date, entry_price,
                exit_date, exit_price,
                pnl
            ]

        return operations_df

    #?__________________________________ User methods ___________________________________ #
    def operate(
        self, 
        data: Union[tuple, pd.Series, pd.DataFrame],
    ):
        """
        Extracts the operations from the signals.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to extract the signals from.
        
        Returns:
            - operations_df (pd.DataFrame): A DataFrame containing the operations extracted from the signals.
            - signals_dfs (list): A list of DataFrames containing the signals extracted from the data.
        """
        # ======= I. Extract signals =======
        try:
            signals_dfs = self.predict(data=data)
        
        except Exception as e:
            print(f"Error in the predict method: {e}")
            return None, None
        
        # ======= II. Handle case when signals_dfs is a single DataFrame =======
        if isinstance(signals_dfs, pd.DataFrame):
            signals_dfs = [signals_dfs]
        
        # ======= III. Extract operations from signals =======
        operations_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self.create_operations)(signals_df) for signals_df in signals_dfs
        )

        operations_df = pd.concat(operations_list, ignore_index=True)
        
        # ======= IV. Sort Operations by date =======
        operations_df.sort_values(by=["entry_date"], inplace=True)

        return operations_df, signals_dfs

