import pandas as pd
import numpy as np


def get_volatility(price_series: pd.Series, window: int):

    returns_series = price_series.pct_change().fillna(0)
    volatility_series = returns_series.rolling(window).std() * np.sqrt(window)

    return volatility_series