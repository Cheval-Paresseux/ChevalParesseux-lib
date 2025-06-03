from ..tools import regression_models as reg

import numpy as np
import pandas as pd



#! ==================================================================================== #
#! ====================== Series Tendency Statistics Functions ======================== #
def get_momentum(
    series: pd.Series
) -> float:
    """
    Compute the momentum of a series.
    
    Parameters:
        - series (pd.Series): Input series.
    
    Returns:
        - momentum (float): Momentum value.
    """
    # ======= I. Extract first and last value =======
    first_value = series.iloc[0]
    last_value = series.iloc[-1]
    
    # ======= II. Compute Momentum =======
    momentum = (last_value - first_value) / first_value
    
    return momentum

#*____________________________________________________________________________________ #
def get_Z_momentum(
    series: pd.Series
) -> float:
    """
    Compute the Z-momentum of a series.
    
    Parameters:
        - series (pd.Series): Input series.
    
    Returns:
        - Z_momentum (float): Z-momentum value.
    """
    # ======= I. Compute Momentum =======
    momentum = get_momentum(series)
    
    # ======= II. Compute Standard Deviation of Returns =======
    returns_series = series.pct_change().dropna()
    returns_standard_deviation = np.std(returns_series) * np.sqrt(len(returns_series)) + 1e-8  # Adding a small value to avoid division by zero
    
    # ======= III. Compute Z-Momentum =======
    Z_momentum = momentum / returns_standard_deviation
    
    return Z_momentum

#*____________________________________________________________________________________ #
def get_simple_TempReg(
    series: pd.Series
) -> tuple:
    """
    Compute the simple temporal regression of a series.
    
    Parameters:
        - series (pd.Series): Input series.
    
    Returns:
        - intercept (float): Intercept of the regression.
        - coefficients (np.ndarray): Coefficients of the regression.
        - statistics (dict): Statistics of the regression.
        - residuals (np.ndarray): Residuals of the regression.
    """
    # ======= I. Fit the temporal regression =======
    X = np.arange(len(series))
    model = reg.OLS_regression()
    model.fit(X, series)
    
    # ======= II. Extract the coefficients and statistics =======
    coefficients = model.coefficients
    intercept = model.intercept
    metrics = model.metrics

    return intercept, coefficients, metrics

#*____________________________________________________________________________________ #
def get_quad_TempReg(
    series: pd.Series
) -> tuple:
    """
    Compute the quadratic temporal regression of a series.
    
    Parameters:
        - series (pd.Series): Input series.
    
    Returns:
        - intercept (float): Intercept of the regression.
        - coefficients (np.ndarray): Coefficients of the regression.
        - statistics (dict): Statistics of the regression.
        - residuals (np.ndarray): Residuals of the regression.
    """
    # ======= 1. Fit the temporal regression =======
    X = np.arange(len(series))
    X = np.column_stack((X, X**2))
    model = reg.OLS_regression()
    model.fit(X, series)
    
    # ======= 2. Extract the coefficients and statistics =======
    coefficients = model.coefficients
    intercept = model.intercept
    metrics = model.metrics
    
    return intercept, coefficients, metrics

#*____________________________________________________________________________________ #
def get_OU_estimation(
    series: pd.Series
) -> tuple:
    """
    Compute the Ornstein-Uhlenbeck estimation of a series.
    
    Parameters:
        - series (pd.Series): Input series.
    
    Returns:
        - mu (float): Mean of the series.
        - theta (float): Speed of mean reversion.
        - sigma (float): Volatility of the series.
        - half_life (float): Half-life of the series.
    """
    # ======== I. Initialize series ========
    series_array = np.array(series)
    differentiated_series = np.diff(series_array)
    mu = np.mean(series)
    
    X = series_array[:-1] - mu  # X_t - mu
    y = differentiated_series  # X_{t+1} - X_t

    # ======== II. Perform OLS regression ========
    model = reg.OLS_regression()
    model.fit(X, y)
    
    # ======== III. Extract Parameters ========
    theta = -model.coefficients[0]
    if theta > 0:
        residuals = y - (model.intercept + np.dot(X, model.coefficients))
        sigma = np.sqrt(np.var(residuals) * 2 * theta)
        half_life = np.log(2) / theta
    else:
        theta = 0
        sigma = 0
        half_life = 0

    return mu, theta, sigma, half_life

#*____________________________________________________________________________________ #
def get_autocorrelation(
    series: pd.Series, 
    lags: int
) -> np.ndarray:
    """
    Compute autocorrelations up to given number of lags.
    
    Parameters:
        - series (pd.Series): Time series data.
        - lags (int): Number of lags to compute.
    
    Returns:
        - np.ndarray: Autocorrelation values for each lag.
    """
    # ======= I. Standardize the series =======
    standardized_series = (series - series.mean()) / series.std()
    
    # ======= II. Compute autocorrelations =======
    result = [] 
    for lag in range(1, lags + 1):
        acf_lag = np.corrcoef(standardized_series[lag:], standardized_series[:-lag])[0, 1]
        result.append(acf_lag)
    
    # ======= III. Transform into np.array =======
    result = np.array(result)

    return result

#*____________________________________________________________________________________ #
def get_partial_autocorrelation(
    series: pd.Series, 
    lags: int
) -> np.ndarray:
    """
    Compute partial autocorrelation using linear regression (OLS) method.
    
    Parameters:
        - series (pd.Series): Time series data.
        - lags (int): Number of lags to compute.
    
    Returns:
        - np.ndarray: Partial autocorrelation values for each lag.
    """
    # ======= I. Iterate through lags =======
    pacf = []  
    for lag in range(1, lags + 1):
        # ----- 1. Create lagged DataFrame -----
        df = pd.DataFrame({'y': series[lag:]})
        for i in range(1, lag + 1):
            df[f'lag_{i}'] = series.shift(i)[lag:]

        # ----- 2. Extract explanatory and target series -----
        df = df.dropna()
        X = df.drop(columns='y').values
        y = df['y'].values

        # ----- 3. Fit OLS regression -----
        reg_model = reg.OLS_regression()
        reg_model.fit(X, y)
        last_coeff = reg_model.coefficients[-1] # Last coefficient is PACF at lag k
        pacf.append(last_coeff)  
    
    # ======= II. Transform into np.array =======
    pacf = np.array(pacf)

    return pacf

#*____________________________________________________________________________________ #
def get_kama(
    series: pd.Series,
    fastest_window: int,
    slowest_window: int,
) -> float:
    """
    Computes the Kaufman Adaptive Moving Average (KAMA) for the last two points in a series.
    
    Parameters:
        - series (pd.Series): The input series containing price data.
        - fastest_window (int): The window size for the fastest smoothing constant.
        - slowest_window (int): The window size for the slowest smoothing constant.
    
    Returns:
        - float: The KAMA value for the last point in the series.
    """
    # ======= I. Inputs =======  
    slowest_window = min(slowest_window, len(series) - 2)
    
    fast_sc = 2 / (fastest_window + 1)
    slow_sc = 2 / (slowest_window + 1)
    
    # ======= II. Compute KAMA value for t-1 =======
    change_t0 = abs(series.iloc[-2] - series.iloc[-2 - slowest_window])
    vol_t0 = series.diff().abs().iloc[-2 - slowest_window + 1 : -1].sum()
    efficiency_ratio_t0 = change_t0 / (vol_t0 + 1e-8)

    smoothing_constant_t0 = (efficiency_ratio_t0 * (fast_sc - slow_sc) + slow_sc) ** 2
    kama_t0 = series.iloc[-3] + smoothing_constant_t0 * (series.iloc[-2] - series.iloc[-3])

    # ======= III. Compute KAMA value for t =======
    change_t1 = abs(series.iloc[-1] - series.iloc[-1 - slowest_window])
    vol_t1 = series.diff().abs().iloc[-1 - slowest_window + 1 :].sum()
    efficiency_ratio_t1 = change_t1 / (vol_t1 + 1e-8)

    smoothing_constant_t1 = (efficiency_ratio_t1 * (fast_sc - slow_sc) + slow_sc) ** 2
    kama_t1 = kama_t0 + smoothing_constant_t1 * (series.iloc[-1] - kama_t0)

    return kama_t1

#*____________________________________________________________________________________ #
def get_relative_strength_index(
    series: pd.Series
) -> float:
    """
    Computes the Relative Strength Index (RSI) for a given price series.
    
    Parameters:
        - series (pd.Series): Price series to compute RSI on.
    
    Returns:
        - float: The RSI value for the last point in the series.
    """
    # ======= I. Compute Gain and Loss =======
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # ======= II. Compute Average Gain and Loss =======
    avg_gain = gain.rolling(window=len(series)).mean()
    avg_loss = loss.rolling(window=len(series)).mean()

    # ======= III. Compute Relative Strength and RSI =======
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = rs / (1 + rs)
    
    return rsi.iloc[-1]

#*____________________________________________________________________________________ #
def get_stochastic_rsi(
    series: pd.Series
) -> float:
    """
    Computes the Stochastic RSI for a given price series.
    
    Parameters:
        - series (pd.Series): Price series to compute Stochastic RSI on.
    
    Returns:
        - float: The Stochastic RSI value for the last point in the series.
    """
    # ========== 0. Define a function to compute the Stochastic RSI =======
    def get_relative_strength_index(
        series: pd.Series
    ) -> pd.Series:
        """
        Computes the Relative Strength Index (RSI) for a given price series.
        
        Parameters:
            - series (pd.Series): Price series to compute RSI on.
        
        Returns:
            - pd.Series: The RSI values for the input series.
        """
        # ======= I. Compute Gain and Loss =======
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # ======= II. Compute Average Gain and Loss =======
        avg_gain = gain.rolling(window=int(len(series) / 2)).mean()
        avg_loss = loss.rolling(window=int(len(series) / 2)).mean()

        # ======= III. Compute Relative Strength and RSI =======
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi
    
    # ======= I. Compute the Relative Strength Index (RSI) =======
    rsi = get_relative_strength_index(series)

    # ======= II. Extract last few RSI values to get the range for StochRSI =======
    rsi_values = rsi.dropna()
    if len(rsi_values) == 0:
        return np.nan

    last_rsi = rsi_values.iloc[-1]
    min_rsi = rsi_values.min()
    max_rsi = rsi_values.max()

    # ======= III. Compute Stochastic RSI =======
    stoch_rsi = (last_rsi - min_rsi) / (max_rsi - min_rsi + 1e-8)
    
    return stoch_rsi

#*____________________________________________________________________________________ #
def get_ehlers_fisher_transform(
    series_high: pd.Series, 
    series_low: pd.Series, 
) -> float:
    """
    Computes the Ehlers Fisher Transform for a given high and low price series.
    
    Parameters:
        - series_high (pd.Series): High price series.
        - series_low (pd.Series): Low price series.
    
    Returns:
        - float: The Fisher Transform value for the last point in the series.
    """
    # ======= I. Compute the mid-series =======
    mid_series = (series_high + series_low) / 2

    # ======= II. Normalize the mid-series to [-1, 1] =======
    min_val = mid_series.min()
    max_val = mid_series.max()
    if max_val - min_val == 0:
        return 0.0  # avoid division by zero

    normalized = 2 * ((mid_series - min_val) / (max_val - min_val)) - 1
    normalized = np.clip(normalized, -0.999, 0.999)

    # ======= III. Compute the Fisher Transform =======
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))

    return fisher.iloc[-1]  # Only return the last value

#*____________________________________________________________________________________ #
def get_oscillator(
    series: pd.Series
) -> float:
    """
    Computes the oscillator value for a given series.
    
    Parameters:
        - series (pd.Series): Input series.
    
    Returns:
        - float: The oscillator value for the last point in the series.
    """
    # ======= I. Compute the linear regression slope =======
    linear_reg = np.polyfit(np.arange(len(series)), series, 1)
    slope = linear_reg[0]
    
    # ======= II. Compute the mean and standard deviation =======
    mean = series.mean()
    stdev = series.std()
    
    # ======= III. Compute the oscillator value =======
    last_value = series.iloc[-1]
    oscillator_value = (last_value - (mean + slope * (len(series) - 1) /2)) / (stdev + 1e-8)
    
    return oscillator_value

#*____________________________________________________________________________________ #
def get_vortex(
    series_mid: pd.Series,
    series_high: pd.Series,
    series_low: pd.Series,
) -> float:
    """
    Computes the Vortex Indicator for a given mid, high, and low price series.
    
    Parameters:
        - series_mid (pd.Series): Mid price series.
        - series_high (pd.Series): High price series.
        - series_low (pd.Series): Low price series.
    
    Returns:
        - tuple: Vortex Up and Vortex Down values.
    """
    # ======= I. Compute the True Range =======
    true_range = np.maximum(
        series_high - series_low,
        np.abs(series_high - series_mid.shift(1)),
        np.abs(series_low - series_mid.shift(1))
    )
    
    # ======= II. Compute the Vortex Movement =======
    vm_up = np.abs(series_high - series_low.shift(1))
    vm_down = np.abs(series_low - series_high.shift(1))
    
    # ======= III. Compute the Vortex Indicator =======
    vortex_up = vm_up.sum() / true_range.sum() if true_range.sum() != 0 else 0.0
    vortex_down = vm_down.sum() / true_range.sum() if true_range.sum() != 0 else 0.0
    
    # ======= IV. Ensure no outliers =======
    if np.abs(vortex_up) > 3 or np.abs(vortex_down) > 3:
        vortex_up, vortex_down = 1, 1

    return vortex_up, vortex_down

#*____________________________________________________________________________________ #
def get_vigor(
    series_open: pd.Series,
    series_close: pd.Series,
    series_high: pd.Series,
    series_low: pd.Series,
) -> float:
    """
    Computes the Vigor Index for a given open, close, high, and low price series.
    
    Parameters:
        - series_open (pd.Series): Open price series.
        - series_close (pd.Series): Close price series.
        - series_high (pd.Series): High price series.
        - series_low (pd.Series): Low price series.
    
    Returns:
        - float: The Vigor Index value for the last point in the series.
    """
    # ======= I. Compute vigor index =======
    vigor = (series_close - series_open) / (series_high - series_low + 1e-8)
    vigor_index = vigor.iloc[-1]  # Return only the last value
    
    return vigor_index

#*____________________________________________________________________________________ #
def get_stochastic_oscillator(
    series_mid: pd.Series,
    series_high: pd.Series,
    series_low: pd.Series,
) -> tuple[float, float]:
    """
    Compute fast (%K) and slow (%D) stochastic oscillator values.

    Parameters:
        - series_mid: pd.Series of close/mid prices
        - series_high: pd.Series of high prices
        - series_low: pd.Series of low prices

    Returns:
        - last_fast_oscillator: last %K value (float)
        - slow_oscillator: mean of last 3 %K values (float)
    """
    # ======= I. Compute rolling %K values =======
    lowest_low = series_low.min()
    highest_high = series_high.max()
    
    # ======= II. Compute %K series =======
    range_ = highest_high - lowest_low + 1e-8
    k_series = (series_mid - lowest_low) / range_

    # ======= III. Extract last fast K and slow D =======
    last_fast_oscillator = k_series.iloc[-1]
    slow_oscillator = k_series.iloc[-3:].mean()  # mean of last 3 %K

    return last_fast_oscillator, slow_oscillator

#*____________________________________________________________________________________ #
