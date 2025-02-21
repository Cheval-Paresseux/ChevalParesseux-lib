import numpy as np


def standard_wiener_simulation(
    X0: float,
    maturity: int,
    nb_observations: int,
):
    """
    Simulate a standard Wiener process.

    Parameters:
        X0 (float): Initial value
        maturity (int): Maturity of the process
        nb_observations (int): Number of observations

    Returns:
        X (np.array): Simulated Wiener
    """
    dt = maturity / nb_observations
    dW = np.sqrt(dt) * np.random.randn(nb_observations)
    W = np.cumsum(dW)

    X = X0 * np.exp(W)

    return X


def general_wiener_simulation(
    X0: float,
    maturity: int,
    nb_observations: int,
    drift: float,
    volatility: float,
):
    """
    Simulate a general Wiener process.

    Parameters:
        X0 (float): Initial value
        maturity (int): Maturity of the process
        nb_observations (int): Number of observations
        drift (float): Drift
        volatility (float): Volatility

    Returns:
        X (np.array): Simulated Wiener
    """
    dt = maturity / nb_observations
    dx = drift * dt + volatility * np.sqrt(dt) * np.random.randn(nb_observations)
    W = np.cumsum(dx)

    X = X0 * np.exp(W)

    return X


def geometric_brownian_motion_simulation(
    X0: float,
    drift: float,
    volatility: float,
    total_time: int,
    number_of_observations: int,
):
    """
    Simulate a geometric Brownian motion (GBM) to model stock prices.

    Parameters:
        X0 (float): Initial stock price
        drift (float): Expected return (drift, mu)
        volatility (float): Volatility of the stock (sigma)
        total_time (float): Total time (in years, or other unit of time)
        number_of_obersvations (int): Number of time steps

    Returns:
        prices (np.array) Simulated stock prices over time
    """
    # ====== I. Initialize the series ======
    dt = total_time / number_of_observations
    prices = np.zeros(number_of_observations)
    prices[0] = X0

    # ====== II. Simulate the GBM process ======
    random_shocks = np.random.normal(0, 1, number_of_observations)
    for t in range(1, number_of_observations):
        prices[t] = prices[t - 1] * np.exp(
            (drift - 0.5 * volatility**2) * dt
            + volatility * np.sqrt(dt) * random_shocks[t]
        )

    return prices


def correlated_geometric_brownian_motion_simulation(
    S0_1: float,
    S0_2: float,
    maturity: int,
    nb_observations: int,
    mu_1: float,
    mu_2: float,
    sigma_1: float,
    sigma_2: float,
    rho: float,
):
    """
    Simulate two correlated geometric Brownian motions.

    Parameters:
        S0_1 (float): Initial value of the first asset
        S0_2 (float): Initial value of the second asset
        maturity (int): Maturity of the process
        nb_observations (int): Number of observations
        mu_1 (float): Drift of the first asset
        mu_2 (float): Drift of the second asset
        sigma_1 (float): Volatility of the first asset
        sigma_2 (float): Volatility of the second asset
        rho (float): Correlation coefficient between the two assets

    Returns:
        S1 (np.array): Simulated prices of the first asset
        S2 (np.array): Simulated prices of the second
    """
    # ====== I. Initialize the series ======
    dt = maturity / nb_observations

    S1 = np.zeros(nb_observations)
    S1[0] = S0_1
    S2 = np.zeros(nb_observations)
    S2[0] = S0_2

    # ====== II. Simulate the correlated Wiener Processes ======
    W1 = np.sqrt(dt) * np.random.randn(nb_observations)
    W2 = rho * W1 + np.sqrt(1 - rho**2) * np.sqrt(dt) * np.random.randn(nb_observations)

    # ====== III. Simulate the correlated GBM process ======
    for i in range(1, nb_observations):
        deltaS1 = mu_1 * S1[i - 1] * dt + sigma_1 * S1[i - 1] * W1[i]
        S1[i] = S1[i - 1] + deltaS1

        deltaS2 = mu_2 * S2[i - 1] * dt + sigma_2 * S2[i - 1] * W2[i]
        S2[i] = S2[i - 1] + deltaS2

    return S1, S2


def multi_correlated_geometric_brownian_motion_simulation(
    S0: np.array,
    mu: np.array,
    sigma: np.array,
    rho: np.array,
    maturity: int,
    nb_observations: int,
):
    """
    Simulate two correlated geometric Brownian motions.

    Parameters:
        S0 (np.array): Initial values of the assets
        maturity (int): Maturity of the process
        nb_observations (int): Number of observations
        mu (np.array): Drift of the assets
        sigma (np.array): Volatility of the assets
        rho (np.array): Correlation coefficients between the assets

    Returns:
        list_series (list): Simulated prices of the assets
    """
    dt = maturity / nb_observations

    # ====== I. Initialize the series ======
    S_1 = np.zeros(nb_observations)
    S_1[0] = S0[0]
    W_1 = np.sqrt(dt) * np.random.randn(nb_observations)
    for i in range(1, nb_observations):
        deltaS_1 = mu[0] * S_1[i - 1] * dt + sigma[0] * S_1[i - 1] * W_1[i]
        S_1[i] = S_1[i - 1] + deltaS_1

    # ====== II. Simulate the correlated GBM process ======
    list_series = [S_1]
    for serie in range(1, len(S0)):
        S = np.zeros(nb_observations)
        S[0] = S0[serie]
        W = rho[serie] * W_1 + np.sqrt(1 - rho[serie] ** 2) * np.sqrt(
            dt
        ) * np.random.randn(nb_observations)
        for i in range(1, nb_observations):
            deltaS = mu[serie] * S[i - 1] * dt + sigma[serie] * S[i - 1] * W[i]
            S[i] = S[i - 1] + deltaS

        list_series.append(S)

    return list_series


def fractional_brownian_motion_simulation(
    H: float,
    maturity: float,
    nb_observations: int,
    X0: float,
    drift: float,
    volatility: float,
):
    """
    Simulate a fractional Brownian motion and the price history of a financial asset.

    Parameters:
        H (float): Hurst parameter
        maturity (float): Maturity of the process
        nb_observations (int): Number of observations
        X0 (float): Initial price
        drift (float): Drift term
        volatility (float): Volatility term

    Returns:
        prices (np.array): Simulated prices of the financial asset
        W (np.array): Simulated fractional Brownian motion
    """
    times = np.linspace(0, maturity, nb_observations)

    # ====== I. Construct covariance matrix ======
    covariance_matrix = np.zeros((nb_observations, nb_observations))
    for i in range(nb_observations):
        for j in range(nb_observations):
            covariance_matrix[i, j] = 0.5 * (
                times[i] ** (2 * H)
                + times[j] ** (2 * H)
                - abs(times[i] - times[j]) ** (2 * H)
            )

    # Add jitter to the diagonal to ensure positive definiteness
    jitter = 1e-10
    covariance_matrix += jitter * np.eye(nb_observations)

    # ====== II. Cholesky decomposition ======
    L = np.linalg.cholesky(covariance_matrix)

    # ====== III. Generate standard normal increments ======
    Z = np.random.randn(nb_observations)

    # ====== IV. Compute the fractional Brownian motion ======
    W = np.dot(L, Z)

    # ====== V. Simulate the price history ======
    prices = X0 * np.exp((drift - 0.5 * volatility**2) * times + volatility * W)

    return prices


def ornstein_uhlenbeck_simulation(
    X0: float,
    mu: float,
    theta: float,
    sigma: float,
    maturity: float,
    nb_observations: int,
):
    """
    Simulate an Ornstein-Uhlenbeck process.

    Parameters:
        X0 (float): Initial value
        mu (float): Mean reversion level
        theta (float): Mean reversion speed
        sigma (float): Volatility
        maturity (float): Maturity of the process
        nb_observations (int): Number of observations

    Returns:
        X (np.array): Simulated Ornstein-Uhlenbeck process
    """
    dt = maturity / nb_observations
    X = np.zeros(nb_observations)
    X[0] = X0

    for t in range(1, nb_observations):
        dW = np.random.randn() * np.sqrt(dt)
        X[t] = X[t - 1] + theta * (mu - X[t - 1]) * dt + sigma * dW

    return X


def power_law_simulation(
    S0: float,
    alpha: float,
    maturity: int,
    nb_observations: int,
    volatility: float,
):
    """
    Simulates a stock price path using power law-distributed returns.

    Parameters:
        S0 (float): The initial stock price.
        alpha (float): The exponent for the power law distribution (alpha > 1).
        T (int): The total time (e.g., 1 year).
        n (int): The number of time steps.
        volatility (float): Volatility scaling factor to control the magnitude of returns.

    Returns:
        np.ndarray: Simulated stock prices.
    """
    # Time increment
    dt = maturity / nb_observations

    # Generate random returns from a power law distribution (Pareto), shifted to have both positive and negative returns
    pareto_returns = np.random.pareto(alpha, nb_observations) - 1
    returns = (
        pareto_returns * volatility * np.random.choice([-1, 1], size=nb_observations)
    )

    # Cumulative log returns
    cumulative_log_returns = np.cumsum(returns) * np.sqrt(dt)

    # Compute stock prices from cumulative log returns
    stock_prices = S0 * np.exp(cumulative_log_returns)

    return stock_prices


def power_law_hurst_simulation(
    S0: float,
    alpha: float,
    H: float,
    maturity: float,
    nb_observations: int,
    volatility: float,
    drift: float,
) -> np.ndarray:
    """
    Simulates a stock price path using power-law distribution for returns
    and incorporates the Hurst exponent for long-term memory.

    Parameters:
        S0 (float): The initial stock price.
        alpha (float): The exponent for the power-law distribution (alpha > 1).
        H (float): The Hurst exponent (0 < H < 1).
        maturity (float): The total time period for simulation.
        nb_observations (int): The number of increments to simulate.
        volatility (float): Volatility scaling factor for power-law returns.
        drift (float): Drift term for the price simulation.

    Returns:
        np.ndarray: Simulated stock prices.
    """
    dt = maturity / nb_observations

    # Simulate fractional Brownian motion based on the Hurst exponent
    fgn = fractional_brownian_motion_simulation(
        H, maturity, nb_observations, S0, drift, volatility
    )

    # Calculate the log returns from the simulated prices
    fgn_returns = np.diff(np.log(fgn))

    # Generate random returns from a power law distribution (Pareto)
    pareto_returns = np.random.pareto(alpha, nb_observations - 1) - 1  # Adjust length
    returns = (
        pareto_returns
        * volatility
        * np.random.choice([-1, 1], size=nb_observations - 1)
    )

    # Combine fractional Brownian motion increments with power-law returns
    combined_returns = fgn_returns + returns

    # Cumulative log returns
    cumulative_log_returns = np.cumsum(combined_returns) * np.sqrt(dt)

    # Compute stock prices from cumulative log returns
    stock_prices = S0 * np.exp(cumulative_log_returns)

    return stock_prices
