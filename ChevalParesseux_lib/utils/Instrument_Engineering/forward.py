import numpy as np
import pandas as pd
from typing import Union, Self, List
from datetime import datetime

#! ==================================================================================== #
#! ================================= Base Model ======================================= #
class Futures_Pricer:
    def __init__(self, n_jobs: int) -> None:
        self.n_jobs = n_jobs
        self.params = {}
        self.roll_params = {}

    def set_params(
        self,
        maturity: Union[int, str, datetime, pd.Timestamp],
        fix_risk_free: float = None
    ) -> Self:
        self.params = {
            'maturity': maturity,
            'fix_risk_free': fix_risk_free
        }
        return self

    def set_roll_params(
        self,
        contract_expiries: List[Union[str, datetime, pd.Timestamp]],
        roll_days_before_expiry: int
    ) -> Self:
        self.roll_params = {
            'contract_expiries': pd.to_datetime(contract_expiries),
            'roll_days_before_expiry': roll_days_before_expiry
        }
        return self

    def compute_futures_price(
        self,
        spot_price: float,
        risk_free: float,
        time_to_maturity: float
    ) -> float:
        return spot_price * np.exp(risk_free * time_to_maturity)

    def get_futures_price_no_roll(
        self,
        data: pd.DataFrame
    ) -> pd.Series:
        maturity = self.params['maturity']
        risk_free = self.params.get('fix_risk_free', 0.0)

        if isinstance(maturity, (str, datetime)):
            maturity = pd.to_datetime(maturity)

        df = data.copy()
        if "date" not in df.columns:
            raise ValueError("Data must contain a 'date' column.")
        df['date'] = pd.to_datetime(df['date'])

        if isinstance(maturity, (datetime, pd.Timestamp)):
            df['ttm'] = (maturity - df['date']).dt.days / 365
        else:
            df['ttm'] = maturity

        df['futures_price'] = df.apply(
            lambda row: self.compute_futures_price(
                spot_price=row['spot'],
                risk_free=risk_free,
                time_to_maturity=row['ttm']
            ),
            axis=1
        )
        return df[['date', 'futures_price']]

    def get_futures_price_with_roll(
        self,
        data: pd.DataFrame,
        method: Literal['back', 'forward'] = 'back'
    ) -> pd.Series:
        if not self.roll_params:
            raise ValueError("Roll parameters must be set via `set_roll_params()`.")

        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])

        risk_free = self.params.get('fix_risk_free', 0.0)
        expiries = self.roll_params['contract_expiries']
        roll_days = self.roll_params['roll_days_before_expiry']

        df.set_index('date', inplace=True)
        futures_series = pd.Series(index=df.index, dtype=float)

        prev_adj = 0  # For back-adjustment
        prev_price = None

        for i in range(len(expiries) - 1):
            expiry = expiries[i]
            next_expiry = expiries[i + 1]
            roll_date = expiry - pd.Timedelta(days=roll_days)

            mask = (df.index > roll_date) & (df.index <= next_expiry)
            ttm = (expiry - df.index[mask]).days / 365

            prices = df.loc[mask, 'spot'] * np.exp(risk_free * ttm)
            prices = prices.rename("futures_price")

            if prev_price is not None:
                price_diff = prices.iloc[0] - prev_price
                if method == 'back':
                    adj_prices = prices - price_diff
                    prev_adj += -price_diff
                else:  # forward
                    adj_prices = prices + prev_adj
                    prev_adj += price_diff
            else:
                adj_prices = prices
            futures_series.loc[mask] = adj_prices
            prev_price = prices.iloc[-1]

        futures_series.name = 'futures_price'
        return futures_series.reset_index()

