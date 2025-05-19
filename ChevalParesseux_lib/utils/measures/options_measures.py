import pandas as pd
import numpy as np
import scipy.stats as stats



#! ==================================================================================== #
#! ============================ European Options Pricing ============================== #
def get_european_BSM(
    spot: float, 
    strike: float, 
    maturity: float, 
    risk_free_rate: float, 
    volatility: float,
) -> tuple:
    """
    Computes the price of a European option using the Black-Scholes-Merton formula.

    Args: 
        spot (float): The current price of the underlying asset.
        strike (float): The strike price of the option.
        maturity (float): Time to maturity (in years).
        risk_free_rate (float): The annual risk-free interest rate.
        volatility (float): The annual volatility of the underlying asset.

    Returns:
        tuple: (call_price, put_price)
            call_price (float): Price of the European call option.
            put_price (float): Price of the European put option.
    """
    # ======= I. Check inputs =======
    if spot <= 0 or strike <= 0 or maturity <= 0 or volatility <= 0:
        raise ValueError("All inputs must be strictly positive.")

    # ======= II. Calculate d1 and d2 =======
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * maturity) / (volatility * np.sqrt(maturity))
    d2 = d1 - volatility * np.sqrt(maturity)

    # ======= III. Calculate N(d1), N(d2), N(-d1), N(-d2) =======
    N_d1 = stats.norm.cdf(d1)
    N_d2 = stats.norm.cdf(d2)
    N_minus_d1 = stats.norm.cdf(-d1)
    N_minus_d2 = stats.norm.cdf(-d2)

    # ======= IV. Calculate call and put prices =======
    call_price = spot * N_d1 - strike * np.exp(-risk_free_rate * maturity) * N_d2
    put_price = strike * np.exp(-risk_free_rate * maturity) * N_minus_d2 - spot * N_minus_d1

    return call_price, put_price

#*____________________________________________________________________________________ #
def get_european_biTree(
    spot: float, 
    strike: float, 
    maturity: float, 
    risk_free: float, 
    volatility: float,
    number_of_branch: int, 
    up: float = 0,
    down: float = 0
) -> tuple:
    """
    Computes the price of a European option using a binomial tree model.

    Parameters:
        spot (float): Current price of the underlying asset.
        strike (float): Strike price of the option.
        maturity (float): Time to maturity in years.
        risk_free (float): Risk-free interest rate (annualized).
        volatility (float): Annualized volatility of the underlying.
        number_of_branch (int): Number of steps in the binomial tree.
        up (float, optional): Optional up factor (e.g., 0.1 for +10%). If 0, it is derived from volatility.
        down (float, optional): Optional down factor (e.g., 0.1 for -10%). If 0, it is derived from volatility.

    Returns:
        tuple: (call_price, put_price)
    """
    # ======= I. Check inputs =======
    if spot <= 0 or strike <= 0 or maturity <= 0 or number_of_branch <= 0:
        raise ValueError("Spot, strike, maturity, and number of branches must be positive.")
    
    dt = maturity / number_of_branch

    # ======= II. Compute up/down factors =======
    if up > 0 and down > 0:
        u = 1 + up
        d = 1 - down
    else:
        u = np.exp(volatility * np.sqrt(dt))
        d = 1 / u

    # ======= III. Risk-neutral probability =======
    disc = np.exp(-risk_free * dt)
    p = (np.exp(risk_free * dt) - d) / (u - d)
    if not (0 < p < 1):
        raise ValueError(f"Risk-neutral probability out of bounds: p = {p}")

    # ======= IV. Terminal payoffs =======
    call_values = np.zeros(number_of_branch + 1)
    put_values = np.zeros(number_of_branch + 1)

    for j in range(number_of_branch + 1):
        price = spot * (u ** (number_of_branch - j)) * (d ** j)
        call_values[j] = max(0, price - strike)
        put_values[j] = max(0, strike - price)

    # ======= V. Backward induction =======
    for i in range(number_of_branch - 1, -1, -1):
        for j in range(i + 1):
            call_values[j] = disc * (p * call_values[j] + (1 - p) * call_values[j + 1])
            put_values[j] = disc * (p * put_values[j] + (1 - p) * put_values[j + 1])

    # ======= VI. Return the option prices =======
    call_price = call_values[0]
    put_price = put_values[0]

    return call_price, put_price

#*____________________________________________________________________________________ #
def get_IV_interpolation(
    spot: float, 
    strike: float, 
    maturity: float, 
    risk_free_rate: float, 
    call_price: float = 0,
    put_price: float = 0, 
    convergence_threshold: float = 1e-6, 
    max_iterations: int = 1000
) -> tuple:
    """
    Computes the implied volatility of European call and put options using linear interpolation.

    Args:
        call_price (float): Market price of the call option, default is 0.
        put_price (float): Market price of the put option, default is 0.
        spot (float): Current price of the underlying asset.
        strike (float): Option strike price.
        maturity (float): Time to maturity in years.
        risk_free_rate (float): Continuously compounded risk-free interest rate.
        convergence_threshold (float): Desired accuracy of interpolation.
        max_iterations (int): Max number of iterations allowed.

    Returns:
        tuple: (call_vol, put_vol) implied volatilities.
    """
    # ======= I. Initialize variables =======
    # I.1 volatility bounds
    vol_min, vol_max = 1e-5, 5.0
    iteration = 0

    # I.2 initial call and put prices
    call_price_min, put_price_min = get_european_BSM(spot, strike, maturity, risk_free_rate, vol_min)
    call_price_max, put_price_max = get_european_BSM(spot, strike, maturity, risk_free_rate, vol_max)
    call_vol_mid, put_vol_mid = None, None

    # ====== II. Iterative interpolation =======
    while iteration < max_iterations:
        # ---- 1. Compute the min_max range ----
        call_denominator = call_price_max - call_price_min
        put_denominator = put_price_max - put_price_min

        # ---- 2. Interpolate implied volatilities ----
        call_vol_mid = vol_min + (call_price - call_price_min) * (vol_max - vol_min) / call_denominator
        put_vol_mid  = vol_min + (put_price - put_price_min) * (vol_max - vol_min) / put_denominator

        # ---- 3. Compute new options prices ----
        call_price_mid, _ = get_european_BSM(spot, strike, maturity, risk_free_rate, call_vol_mid)
        _, put_price_mid  = get_european_BSM(spot, strike, maturity, risk_free_rate, put_vol_mid)

        # ---- 4. Check convergence ----
        call_error = abs(call_price_mid - call_price)
        put_error  = abs(put_price_mid - put_price)

        if call_error < convergence_threshold and put_error < convergence_threshold:
            return call_vol_mid, put_vol_mid

        # ---- 5. Update bounds ----
        # 5.1 Call option
        if call_price_mid < call_price:
            vol_min = call_vol_mid
            call_price_min = call_price_mid
        else:
            vol_max = call_vol_mid
            call_price_max = call_price_mid

        # 5.2 Put option
        if put_price_mid < put_price:
            vol_min = min(vol_min, put_vol_mid)
            put_price_min = put_price_mid
        else:
            vol_max = max(vol_max, put_vol_mid)
            put_price_max = put_price_mid

        iteration += 1

    return call_vol_mid, put_vol_mid





