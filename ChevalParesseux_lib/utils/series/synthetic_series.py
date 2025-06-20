import pandas as pd
import numpy as np
from typing import Tuple, List


#! ==================================================================================== #
#! =========================== Series Decompositon Functions ========================== #
def get_series_decomposition(
    data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    """
    Decomposes a time series into intraday, overnight, and complete returns.

    Parameters:
        - data (pd.DataFrame): DataFrame with columns ['date', 'open', 'close', 'high', 'low'].

    Returns:
        - intraday_df (pd.DataFrame): DataFrame with intraday returns and related metrics.
        - overnight_df (pd.DataFrame): DataFrame with overnight returns and related metrics.
        - complete_df (pd.DataFrame): DataFrame with complete returns and related metrics.
        - daily_data (List[pd.DataFrame]): List of DataFrames for each day.
    """
    # ======= 0. Helper Functions =======
    def get_returns(
        open_price: float, 
        close_price: float, 
        high_price: float, 
        low_price: float
    ) -> Tuple[float, float, float]:
        """
        Computes the different returns based on open, close, high, and low prices.
        
        Parameters:
            open_price (float): The opening price of the asset.
            close_price (float): The closing price of the asset.
            high_price (float): The highest price of the asset during the period.
            low_price (float): The lowest price of the asset during the period.
        
        Returns:
            Tuple[float, float, float]: A tuple containing:
                - Return based on close price
                - Return based on high price
                - Return based on low price
        """
        # ======= I. Validate Inputs =======
        if open_price == 0 or pd.isna(open_price):
            return np.nan, np.nan, np.nan
        
        # ======= II. Computation =======
        close_return = (close_price - open_price) / open_price
        high_return = (high_price - open_price) / open_price
        low_return = (low_price - open_price) / open_price
        
        return close_return, high_return, low_return
    
    #?____________________________________________________________________________________ #
    def reconstruct_prices(
        base_open: float, 
        returns: float, 
        high_return: float, 
        low_return: float
    ) -> Tuple[float, float, float]:
        """
        Reconstructs the close, high, and low prices based on the base open price and returns.
        
        Parameters:
            base_open (float): The base open price to reconstruct from.
            returns (float): The return based on the close price.
            high_return (float): The return based on the high price.
            low_return (float): The return based on the low price.
        
        Returns:
            Tuple[float, float, float]: A tuple containing:
                - Reconstructed close price
                - Reconstructed high price
                - Reconstructed low price
        """
        # ======= I. Apply returns =======
        close = base_open * (1 + returns)
        high = base_open * (1 + high_return)
        low = base_open * (1 + low_return)
        
        return close, high, low
    
    #?____________________________________________________________________________________ #
    
    # ======= I. Pre-process data =======
    daily_data = [group for _, group in data.groupby('date')]
    dates = sorted(data['date'].unique())

    columns = ['returns', 'returns_high', 'returns_low', 'open', 'close', 'high', 'low']
    intraday_df = pd.DataFrame(index=dates, columns=columns, dtype=float)
    overnight_df = pd.DataFrame(index=dates, columns=columns, dtype=float)
    complete_df = pd.DataFrame(index=dates, columns=columns, dtype=float)

    # ======= II. Initialization for day 0 =======
    first_day = daily_data[0]
    second_day = daily_data[1] if len(daily_data) > 1 else None

    # --- 1. Intraday Initialization ---
    intraday_df.iloc[0] = [0, 0, 0, first_day['open'].iloc[0], first_day['close'].iloc[-1], first_day['high'].max(), first_day['low'].min()]

    if second_day is not None:
        # --- 2. Overnight Initialization ---
        overnight_open = first_day['close'].iloc[-1]
        overnight_close = second_day['open'].iloc[0]
        overnight_high = max(overnight_open, overnight_close)
        overnight_low = min(overnight_open, overnight_close)
        overnight_returns = get_returns(overnight_open, overnight_close, overnight_high, overnight_low)
        overnight_df.iloc[0] = [*overnight_returns, overnight_open, overnight_close, overnight_high, overnight_low]

        # --- 3. Complete Initialization ---
        complete_open = first_day['open'].iloc[0]
        complete_close = overnight_close
        complete_high = max(first_day['high'].max(), overnight_close)
        complete_low = min(first_day['low'].min(), overnight_close)
        complete_returns = get_returns(complete_open, complete_close, complete_high, complete_low)
        complete_df.iloc[0] = [*complete_returns, complete_open, complete_close, complete_high, complete_low]

    # ======= III. Main Loop =======
    for i in range(1, len(daily_data) - 1):
        day = daily_data[i]
        date = day['date'].iloc[0]
        next_open = daily_data[i + 1]['open'].iloc[0]

        # --- 1. Intra-day Returns ---
        open_intra = day['open'].iloc[0]
        close_intra = day['close'].iloc[-1]
        high_intra = day['high'].max()
        low_intra = day['low'].min()
        r_intra, r_high, r_low = get_returns(open_intra, close_intra, high_intra, low_intra)
        intraday_df.loc[date, ['returns', 'returns_high', 'returns_low']] = r_intra, r_high, r_low
        intraday_df.loc[date, 'open'] = intraday_df.loc[dates[i-1], 'close']
        intraday_df.loc[date, 'close'], intraday_df.loc[date, 'high'], intraday_df.loc[date, 'low'] = reconstruct_prices(intraday_df.loc[date, 'open'], r_intra, r_high, r_low)

        # --- 2. Overnight Returns ---
        open_o = close_intra
        close_o = next_open
        high_o = max(open_o, close_o)
        low_o = min(open_o, close_o)
        r_o, r_oh, r_ol = get_returns(open_o, close_o, high_o, low_o)
        overnight_df.loc[date, ['returns', 'returns_high', 'returns_low']] = r_o, r_oh, r_ol
        overnight_df.loc[date, 'open'] = overnight_df.loc[dates[i-1], 'close']
        overnight_df.loc[date, 'close'], overnight_df.loc[date, 'high'], overnight_df.loc[date, 'low'] = reconstruct_prices(overnight_df.loc[date, 'open'], r_o, r_oh, r_ol)

        # --- 3. Complete Returns ---
        r_c, r_ch, r_cl = get_returns(open_intra, next_open, max(high_intra, next_open), min(low_intra, next_open))
        complete_df.loc[date, ['returns', 'returns_high', 'returns_low']] = r_c, r_ch, r_cl
        complete_df.loc[date, 'open'] = complete_df.loc[dates[i-1], 'close']
        complete_df.loc[date, 'close'], complete_df.loc[date, 'high'], complete_df.loc[date, 'low'] = reconstruct_prices(complete_df.loc[date, 'open'], r_c, r_ch, r_cl)

    # ======= IV. Final Clean-up =======
    for df in [intraday_df, overnight_df, complete_df]:
        df.drop(columns=['returns_high', 'returns_low'], inplace=True)
        df.dropna(inplace=True)

    return intraday_df, overnight_df, complete_df, daily_data
