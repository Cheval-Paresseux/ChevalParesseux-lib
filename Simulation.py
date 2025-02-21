"""
# Description: This file contains the classes and functions used to simulate the pairs trading strategy.
_____
StatsArb_Portfolio: Class used to generate a portfolio of pairs using the functions of this class.
                    Also contains functions to evaluate the portfolio current state and update it if we change date without needing to recompute everything.
_____
daily_return: Function used to compute the daily return of a portfolio of combinations between yesterday and today.
compute_stats: Function used to compute different statistics and metrics to evaluate a given strategy.
_____
run_simulation: Function used to run a simulation of the trading strategy over a given period of time.
                (WARNING: Likely to be the main source of potential Data Leakage in the codebase)
_____
POTENTIAL IMPROVEMENTS:
    - Improvements in this section come from the other parts of the codebase, it just need be integrated here.
"""

import pandas as pd
import numpy as np
from time import time
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

import os
import sys

sys.path.append(os.path.abspath(".."))
import models.Clustering as Clustering
import models.Information as Information
import models.Portfolio as Portfolio

import warnings

warnings.filterwarnings("ignore")


# ========================================================================================================== #
class StatsArb_Portfolio:
    # =========================================== Portfolio Generation Methods =========================================== #
    def __init__(
        self,
        data_storage: list,
        big_data: pd.DataFrame,
        date: str,
        training_window: int = 365,
    ):
        """
        Args:
            data_storage (list): List of DataFrames, each containing the historical data of a group of assets.
            big_data (pd.DataFrame): DataFrame containing all the prices data of the S&P500 over a large amount of time.
            date (str): The date at which we want to generate the portfolio.
            training_window (int): The number of days used to train the model.
        """
        # ======== I. Storing the input data ========
        self.data_storage = data_storage
        self.big_data = big_data
        self.date = date
        self.training_window = training_window

        self.train_start_date = str(
            pd.to_datetime(self.date) - pd.DateOffset(days=self.training_window)
        )
        self.train_end_date = date

        # -------- We assume that we know at least 10 days in advance when a stock in going to be delisted, so we can remove it from our data sample --------
        self.delisting_check_date = str(
            pd.to_datetime(self.date) + pd.DateOffset(days=10)
        )

        # ======== II. Initializing the training samples ========
        self.train_samples = []
        for group in self.data_storage:
            # -------- 1. Create the train_sample --------
            train_sample = group[group.index >= self.train_start_date]
            train_sample = train_sample[train_sample.index <= self.train_end_date]
            train_sample = train_sample.dropna(axis=1)

            # -------- 2. Remove the delisted stocks --------
            delisting_date = group.index[group.index <= self.delisting_check_date][-1]
            delisted_stocks = group.columns[group.loc[delisting_date].isna()]
            delisted_stocks = delisted_stocks.intersection(train_sample.columns)
            train_sample = train_sample.drop(columns=delisted_stocks)

            self.train_samples.append(train_sample)

        # ======== III. Initializing the intermediate variables of the process ========
        self.clusters_list = []
        self.combinations_list = []
        self.combinations_infos_list = []
        self.filtered_combinations_list = []

        # ======== IV. Initializing the final portfolio and its information ========
        self.portfolio = pd.DataFrame()
        self.capital_engaged = 0
        self.nb_pairs = 0
        self.pf_expected_return = 0
        self.pf_volatility = 0
        self.pf_half_life = 0
        self.pf_sharpe = 0

    # ---------------------------------------------------------------------------------------------------------------------#
    def date_update(self, date: str):
        """
        Update the date of the portfolio and the training samples.
        Args:
            date (str): The new date of the portfolio.

        Returns:
            train_samples (list): List of DataFrames, each containing the training samples of a group of assets.
        """
        # ======== I. Update the dates ========
        self.date = date
        self.train_start_date = str(
            pd.to_datetime(self.date) - pd.DateOffset(days=self.training_window)
        )
        self.train_end_date = date

        self.delisting_check_date = str(
            pd.to_datetime(self.date) + pd.DateOffset(days=10)
        )

        # ======== II. Update the training samples ========
        self.train_samples = []
        for group in self.data_storage:
            train_sample = group[group.index >= self.train_start_date]
            train_sample = train_sample[train_sample.index <= self.train_end_date]
            train_sample = train_sample.dropna(axis=1)

            # -------- 2. Remove the delisted stocks --------
            delisting_date = group.index[group.index <= self.delisting_check_date][-1]
            delisted_stocks = group.columns[group.loc[delisting_date].isna()]
            delisted_stocks = delisted_stocks.intersection(train_sample.columns)
            train_sample = train_sample.drop(columns=delisted_stocks)

            self.train_samples.append(train_sample)

        return self.train_samples

    # ---------------------------------------------------------------------------------------------------------------------#
    def universe_clusters(
        self,
        clustering_method: str = "riskfolio",
        linkage: str = "ward",
        n_clusters: int = 5,
    ):
        """
        Clusterize the universe of assets in order to reduce the number of combinations to test.

        Args:
            clustering_method (str): Method used to clusterize the assets. Possible inputs : "riskfolio" / "dtw".
            linkage (str): Linkage method for the riskfolio method.
            n_clusters (int): Number of clusters for the DTW method.

        Returns:
            clusters_list (list): List of DataFrames, each containing the assets of a cluster.
        """
        # -------- I. Clean the Portfolio clusters information --------
        self.clusters_list = []

        # -------- II. Clusterize the assets --------
        for train_sample in self.train_samples:
            if clustering_method == "riskfolio":
                clusters_list = Clustering.riskfolio_clustering(
                    df=train_sample, linkage=linkage
                )
            elif clustering_method == "dtw":
                clusters_list = Clustering.dtw_clustering(
                    df=train_sample, n_clusters=n_clusters
                )

            # -------- III. Update the Portfolio clusters information --------
            self.clusters_list += clusters_list

        return self.clusters_list

    # ---------------------------------------------------------------------------------------------------------------------#
    def generate_combinations(
        self, assets_per_comb: int = 2, max_shared_assets: int = 6
    ):
        """
        Generate all possible combinations of assets for each cluster.

        Args:
            assets_per_comb (int): The number of assets we want in each combination.
            max_shared_assets (int): The maximum number of times an asset can be shared between different combinations.

        Returns:
            combinations_list (list): List of DataFrames, each containing a combination of assets.
        """
        # ------- I. Generate combinations -------
        self.combinations_list = []

        for cluster in self.clusters_list:
            combinations_list = Information.generate_combinations(
                df=cluster,
                num_assets_per_comb=assets_per_comb,
                max_shared_assets=max_shared_assets,
            )

            self.combinations_list += combinations_list

        return self.combinations_list

    # ---------------------------------------------------------------------------------------------------------------------#
    def get_combination_informations(
        self,
        Zup_exit_threshold: float = 3,
        leverage: bool = False,
        cash_margin: float = 0.2,
        kf_smooth_coefficient: float = 0.8,
        use_kf_weight: bool = True,
        risk_free_rate: float = 0.00035,
        collateralization_level: float = 1.25,
        haircut: float = 0.3,
    ):
        """
        Compute the necessary informations for each combination of assets.

        Args:
            leverage (bool): If True, the portfolio will be leveraged.
            risk_free_rate (float): The risk-free rate used in the computation of the expected return.
            collateralisation_level (float): The collateralisation level required when taking a short position.
            haircut (float): The haircut applied to the cash needed to enter the position.

        Returns:
            combinations_infos (list): List of dictionaries, each containing the informations of a combination.
        """
        # ======== I. Initialize the list of combinations informations ========
        self.combinations_infos_list = []

        # ======== II. Compute the informations for each combination ========
        for combination in self.combinations_list:
            # --------- 0. Name of the pair -------
            assets_name = combination.columns.tolist()
            log_data = np.log(combination)

            # --------- 1. Cointegration test -------
            coefficients_with_intercept, adf_results, kpss_results, residuals = (
                Information.cointegration_test(df=combination)
            )
            weights_copy = coefficients_with_intercept.copy()
            intercept = coefficients_with_intercept[0]

            # --------- 1.1. OU parameters estimation -------
            residuals_array = np.array(residuals)
            mu, theta, sigma, half_life = Information.estimate_ou_parameters(
                data=residuals_array
            )

            if (theta == 0) or (sigma == 0) or (half_life == 0):
                continue

            # --------- 1.2. Kalman Filter estimation -------
            kalman_filter = Information.OU_KalmanFilter(
                mean=mu,
                theta=theta,
                obs_sigma=sigma,
                smooth_coefficient=kf_smooth_coefficient,
            )
            filtered_states, _ = kalman_filter.series_filter(residuals=residuals)

            last_date = filtered_states.index[-1]
            last_kf_value = filtered_states[-1]

            kf_weight_numerator = last_kf_value - intercept
            for i in range(1, len(weights_copy) - 1):
                kf_weight_numerator -= (
                    weights_copy[i] * log_data[assets_name[i - 1]].loc[last_date]
                )
            kf_weight = kf_weight_numerator / log_data[assets_name[-1]].loc[last_date]

            if use_kf_weight:
                weights_copy[-1] = kf_weight
                last_residual = last_kf_value
            else:
                last_residual = residuals.iloc[-1]

            # --------- 2. Compute the normalized weights -------
            total_sum = sum(abs(x) for x in weights_copy[1:])
            normalized_weights = []
            for i in range(1, len(weights_copy)):  # Start from index 1
                weight = weights_copy[i] / total_sum
                normalized_weights.append(weight)

            # --------- 3. Z-score computation -------
            if sigma != 0:
                z_score = (last_residual - mu) / sigma
            else:
                z_score = 0

            # --------- 4. Compute the expected return -------
            leverage = bool(leverage)
            gap = abs(last_residual - mu)

            expected_return, cash_needed, leverage_ratio = (
                Information.get_expected_return(
                    normalized_weights=normalized_weights,
                    z_score=z_score,
                    half_life=half_life,
                    theta=theta,
                    sigma=sigma,
                    mu=mu,
                    gap=gap,
                    Zup_exit_threshold=Zup_exit_threshold,
                    leverage=leverage,
                    risk_free_rate=risk_free_rate,
                    cash_margin=cash_margin,
                    collateralization_level=collateralization_level,
                    haircut=haircut,
                )
            )

            # --------- 6. Store the informations -------
            combination_data = {
                "assets_name": assets_name,
                "normalized_weights": normalized_weights,
                "expected_return": expected_return,
                "cash_needed": cash_needed,
                "leverage_ratio": leverage_ratio,
                "regression_weights": coefficients_with_intercept,
                "kf_weight": kf_weight,
                "z_score": z_score,
                "gap": gap,
                "half_life": half_life,
                "volatility": sigma,
                "mu": mu,
                "theta": theta,
                "adf_pvalue": adf_results[1],
                "kpss_pvalue": kpss_results[1],
                "residuals": residuals,
            }

            # --------- 7. Store the data -------
            self.combinations_infos_list.append(combination_data)

        return self.combinations_infos_list

    # ---------------------------------------------------------------------------------------------------------------------#
    def filter_combinations(
        self,
        adf_pvalue_threshold: float = 0.05,
        kpss_pvalue_threshold: float = 0.05,
        min_ret: float = 0.00175,
        lower_Z_bound: float = 1,
        upper_Z_bound: float = 2,
    ):
        """
        Filter the combinations before applying the portfolio optimization.
        Args:
            adf_threshold (float): The threshold for the ADF test.
            min_ret (float): The minimum expected return of a pair.
            lower_Z_bound (float): The lower bound of the z-score.
            upper_Z_bound (float): The upper bound of the z-score.

        Returns:
            filtered_combinations_list (list): List of dictionaries, each containing the informations of a combination.
        """
        self.filtered_combinations_list = []

        for combination in self.combinations_infos_list:
            if (
                (combination["z_score"] != 0)
                and (abs(combination["z_score"]) < upper_Z_bound)
                and (abs(combination["z_score"]) > lower_Z_bound)
                and (combination["adf_pvalue"] <= adf_pvalue_threshold)
                and (combination["kpss_pvalue"] >= kpss_pvalue_threshold)
                and (combination["expected_return"] > min_ret)
            ):
                self.filtered_combinations_list += [combination]

        return self.filtered_combinations_list

    # ---------------------------------------------------------------------------------------------------------------------#
    def portfolio_optimization(
        self,
        model: str = "TargetVolatility",
        target_volatility: float = 0.2 / np.sqrt(252),
        min_weight: float = 0.03,
        upper_bound: float = 0.25,
        risk_measure: str = "MV",
        objective_function: str = "Sharpe",
    ):
        """
        Generate the portfolio of pairs using the filtered combinations.

        Args:
            model (str): The model used for the portfolio optimization.
            target_volatility (float): The target volatility level, used in the "TargetVolatility" model.
            min_weight (float): The minimum weight a pair should have in the portfolio.
            upper_bound (float): The maximum weight a pair should have in the portfolio.
            risk_measure (str): The risk measure used in the optimization, used in the other models.
            objective_function (str): The objective function used in the optimization, used in the other models.

        Returns:
            portfolio (pd.DataFrame): DataFrame containing the portfolio's informations.
        """
        # ======== I. Optimize the weights for each assets ========
        # --------- 1. Prepare the data -------
        cov_matrix, expected_return_vector, returns_df = (
            Portfolio.covmat_expectret_returnsdf(
                combinations_list=self.filtered_combinations_list
            )
        )

        # --------- 2. Prepare the boundary -------
        N = len(expected_return_vector)
        boundary = np.array([[0, upper_bound]] * N)

        # --------- 3. Compute the optimal weights -------
        if model == "TargetVolatility":
            optimal_weights = Portfolio.targetVolatility_optimization(
                exp_ret=expected_return_vector,
                cov=cov_matrix,
                boundary=boundary,
                target_volatility=target_volatility,
            )

        elif model == "Classic" or model == "RiskParity":
            optimal_weights = Portfolio.riskfolio_optimization(
                returns_df=returns_df,
                cov_matrix=cov_matrix,
                expected_return_vector=expected_return_vector,
                model=model,
                risk_measure=risk_measure,
                objective_function=objective_function,
            )

        elif model == "EqualWeights":
            optimal_weights = Portfolio.equalWeights_optimization(
                returns_df=returns_df,
            )

        elif model == "InverseVolatility":
            optimal_weights = Portfolio.inverseVolatility_optimization(
                cov_matrix=cov_matrix,
            )

        # ======== II. Store the portfolio computed as a DataFrame ========
        # --------- 1. Prepare the DataFrame -------
        optimal_weights_T = optimal_weights.T
        portfolio = pd.DataFrame(
            columns=[
                "assets_name",
                "combination_weights",
                "normalized_weights",
                "expected_return",
                "cash_needed",
                "leverage_ratio",
                "regression_weights",
                "kf_weight",
                "z_score",
                "gap",
                "half_life",
                "volatility",
                "adf_pvalue",
                "kpss_pvalue",
            ]
        )

        # --------- 2. Store the data -------
        x = 0
        for comb_dict in self.filtered_combinations_list:
            combination_weight = optimal_weights_T.iloc[x].values[0]

            if combination_weight > min_weight:
                # --------- i. Extract the data to create the portfolio
                assets_name = comb_dict["assets_name"]
                normalized_weights = comb_dict["normalized_weights"]
                expected_return = comb_dict["expected_return"]
                cash_needed = comb_dict["cash_needed"]
                leverage_ratio = comb_dict["leverage_ratio"]
                regression_weights = comb_dict["regression_weights"]
                kf_weight = comb_dict["kf_weight"]
                z_score = comb_dict["z_score"]
                gap = comb_dict["gap"]
                half_life = comb_dict["half_life"]
                volatility = comb_dict["volatility"]
                adf_pvalue = comb_dict["adf_pvalue"]
                kpss_pvalue = comb_dict["kpss_pvalue"]

                # --------- ii. Store the data
                portfolio.loc[x] = {
                    "assets_name": assets_name,
                    "combination_weights": combination_weight,
                    "normalized_weights": normalized_weights,
                    "expected_return": expected_return,
                    "cash_needed": cash_needed,
                    "leverage_ratio": leverage_ratio,
                    "regression_weights": regression_weights,
                    "kf_weight": kf_weight,
                    "z_score": z_score,
                    "gap": gap,
                    "half_life": half_life,
                    "volatility": volatility,
                    "adf_pvalue": adf_pvalue,
                    "kpss_pvalue": kpss_pvalue,
                }

            x += 1

        # --------- 3. Normalize the weights (min_weight could have cut some weights) -------
        total_weights = portfolio["combination_weights"].sum()
        portfolio["combination_weights"] = (
            portfolio["combination_weights"] / total_weights
        )
        portfolio["combination_weights"] = (
            portfolio["combination_weights"] * portfolio["leverage_ratio"]
        )

        # ======== III. Store the portfolio and the capital engaged ========
        self.portfolio = portfolio
        self.capital_engaged = self.cash_requirements()

        return self.portfolio

    # ---------------------------------------------------------------------------------------------------------------------#
    def round_weights(self, budget: float = 1e6):
        """
        Adjust the weights of the portfolio to match the non fractionnal shares constraint.

        Args:
            budget (float): The total budget of the portfolio.

        Returns:
            rounded_portfolio (pd.DataFrame): DataFrame containing the portfolio's informations with adjusted weights.
        """
        # ======= I. Select the last date of the training period =======
        big_data_cut = self.big_data[self.big_data.index <= self.date]
        date = big_data_cut.index[-1]

        # ======= II. Adjust the weights =======
        rounded_portfolio = self.portfolio.copy()
        for index, row in rounded_portfolio.iterrows():
            pair_budget = budget * row["combination_weights"]

            i = 0
            for asset in row["assets_name"]:
                asset_budget = pair_budget * row["normalized_weights"][i]
                price_asset = self.big_data[asset].loc[date]

                adjusted_asset_budget = int(asset_budget / price_asset) * price_asset
                adjusted_asset_weight = adjusted_asset_budget / pair_budget
                rounded_portfolio.at[index, "normalized_weights"][i] = (
                    adjusted_asset_weight
                )
                i += 1

        self.portfolio = rounded_portfolio
        self.capital_engaged = self.cash_requirements()

        return rounded_portfolio

    # ---------------------------------------------------------------------------------------------------------------------#
    def cash_requirements(self):
        """
        Compute the cash engaged in the portfolio.

        Returns:
            cash_engaged (float): The amount of cash engaged in the portfolio.
        """
        if not self.portfolio.empty:
            cash_engaged = (
                self.portfolio["combination_weights"] * self.portfolio["cash_needed"]
            ).sum()
            self.capital_engaged = cash_engaged
        else:
            cash_engaged = 0

        return cash_engaged

    # ---------------------------------------------------------------------------------------------------------------------#
    def portfolio_evaluation(self):
        """
        Evaluate the portfolio's expected performance.

        Returns:
            results (dict): Dictionary containing the portfolio's statistics.
        """
        # ====== 0. Set up the data and check that portfolio is not empty ======
        start_date = pd.to_datetime(self.date) - pd.DateOffset(
            days=self.training_window
        )
        data = self.big_data.loc[start_date : self.date]

        if self.portfolio.empty:
            self.capital_engaged = 0
            self.nb_pairs = 0
            self.pf_expected_return = 0
            self.pf_volatility = 0
            self.pf_half_life = 0
            self.pf_sharpe = 0

            return {
                "capital_engaged": 0,
                "nb_pairs": 0,
                "pf_expected_return": 0,
                "pf_volatility": 0,
                "pf_half_life": 0,
                "pf_sharpe": 0,
            }

        # ====== I. General Informations ======
        nb_pairs = len(self.portfolio[self.portfolio["combination_weights"] > 0])
        capital_engaged = self.cash_requirements()

        # ====== II. Portfolio Statistics ======
        pf_expected_return = (
            self.portfolio["expected_return"] * self.portfolio["combination_weights"]
        ).sum()

        # ====== III. Portfolio Risk ======
        portfolio_residuals = pd.DataFrame()
        portfolio_residuals.index = data.index
        for index, row in self.portfolio.iterrows():
            assets = row["assets_name"]
            regression_weights = row["regression_weights"][1:]
            intercept = row["regression_weights"][0]
            z_score = row["z_score"]

            log_df = np.log(data[assets])
            residual = np.dot(log_df, regression_weights) + intercept
            if z_score > 0:
                residual = -residual
            portfolio_residuals[f"comb_{index}"] = residual

        portfolio_weights = self.portfolio["combination_weights"].values
        pf_residuals = np.dot(portfolio_residuals, portfolio_weights)
        portfolio_residuals["portfolio_residuals"] = pf_residuals
        portfolio_returns = portfolio_residuals["portfolio_residuals"].diff()

        pf_volatility = portfolio_returns.std()
        pf_half_life = (
            self.portfolio["half_life"] * self.portfolio["combination_weights"]
        ).sum()
        if pf_volatility != 0:
            pf_sharpe = (pf_expected_return * 2 * pf_half_life) / (
                pf_volatility * np.sqrt(2 * pf_half_life)
            )
        else:
            pf_sharpe = 0

        # ====== IV. Store the results ======
        self.capital_engaged = capital_engaged
        self.nb_pairs = nb_pairs
        self.pf_expected_return = pf_expected_return
        self.pf_volatility = pf_volatility
        self.pf_half_life = pf_half_life
        self.pf_sharpe = pf_sharpe

        return {
            "capital_engaged": capital_engaged,
            "nb_pairs": nb_pairs,
            "pf_expected_return": pf_expected_return,
            "pf_volatility": pf_volatility,
            "pf_half_life": pf_half_life,
            "pf_sharpe": pf_sharpe,
        }

    # ---------------------------------------------------------------------------------------------------------------------#
    def generate_portfolio(
        self,
        date: str,
        # Clustering parameters
        clustering_method: str = "riskfolio",
        linkage: str = "ward",
        n_clusters: int = 5,
        # Combinations generation parameters
        assets_per_comb: int = 2,
        max_shared_assets: int = 6,
        # Combination Informations parameters
        Zup_exit_threshold: float = 3,
        leverage: bool = True,
        cash_margin: float = 0.2,
        kf_smooth_coefficient: float = 0.8,
        use_kf_weight: bool = True,
        risk_free_rate: float = 0.00035,
        collateralization_level: float = 1.25,
        haircut: float = 0.3,
        # Filter parameters
        adf_pvalue_threshold: float = 0.05,
        kpss_pvalue_threshold: float = 0.05,
        min_ret: float = 0.00175,
        lower_Z_bound: float = 1,
        upper_Z_bound: float = 2,
        # Portfolio optimization parameters
        model: str = "TargetVolatility",
        target_volatility: float = 0.2 / np.sqrt(252),
        min_weight: float = 0.03,
        upper_bound: float = 0.25,
        risk_measure: str = "MV",
        objective_function: str = "Sharpe",
        # Budget parameters
        budget: float = 1e6,
    ):
        """
        Generate a portfolio of pairs using the functions of this class.

        Args:
            clustering_method (str): Method used to clusterize the assets. Possible inputs : "riskfolio" / "dtw".
            linkage (str): Linkage method for the riskfolio method.
            n_clusters (int): Number of clusters for the DTW method.
            assets_per_comb (int): The number of assets we want in each combination.
            max_shared_assets (int): The maximum number of times an asset can be shared between different combinations.
            leverage (bool): If True, the portfolio will be leveraged.
            cash_margin (float): The percentage of the budget that is not engaged in the portfolio.
            kf_smooth_coefficient (float): The coefficient used in the Kalman Filter.
            use_kf_weight (bool): If True, the Kalman Filter weight will be used in the portfolio.
            risk_free_rate (float): The risk-free rate used in the computation of the expected return.
            collateralisation_level (float): The collateralisation level required when taking a short position.
            haircut (float): The haircut applied to the cash needed to enter the position.
            adf_threshold (float): The threshold for the ADF test.
            min_ret (float): The minimum expected return of a pair.
            lower_Z_bound (float): The lower bound of the z-score.
            upper_Z_bound (float): The upper bound of the z-score.
            model (str): The model used for the portfolio optimization.
            target_volatility (float): The target volatility level.
            min_weight (float): The minimum weight a pair should have in the portfolio.
            upper_bound (float): The maximum weight a pair should have in the portfolio.
            risk_measure (str): The risk measure used in the optimization.
            objective_function (str): The objective function used in the optimization.
            budget (float): The total budget of the portfolio.

        Returns:
            portfolio (pd.DataFrame): DataFrame containing the portfolio's informations.
        """
        # ======= 0. Update the date =======
        self.date_update(date=date)

        # ======= I. Generate the clusters =======
        self.universe_clusters(
            clustering_method=clustering_method, linkage=linkage, n_clusters=n_clusters
        )

        # ======= II. Generate the combinations =======
        self.generate_combinations(
            assets_per_comb=assets_per_comb, max_shared_assets=max_shared_assets
        )

        # ======= III. Compute the combinations informations =======
        self.get_combination_informations(
            Zup_exit_threshold=Zup_exit_threshold,
            leverage=leverage,
            cash_margin=cash_margin,
            kf_smooth_coefficient=kf_smooth_coefficient,
            use_kf_weight=use_kf_weight,
            risk_free_rate=risk_free_rate,
            collateralization_level=collateralization_level,
            haircut=haircut,
        )

        # ======= IV. Filter the combinations =======
        self.filter_combinations(
            adf_pvalue_threshold=adf_pvalue_threshold,
            kpss_pvalue_threshold=kpss_pvalue_threshold,
            min_ret=min_ret,
            lower_Z_bound=lower_Z_bound,
            upper_Z_bound=upper_Z_bound,
        )

        # ======= V. Optimize the portfolio =======
        if len(self.filtered_combinations_list) > 0:
            self.portfolio_optimization(
                model=model,
                target_volatility=target_volatility,
                min_weight=min_weight,
                upper_bound=upper_bound,
                risk_measure=risk_measure,
                objective_function=objective_function,
            )
            self.round_weights(budget=budget)
        else:
            self.portfolio = pd.DataFrame()

        # ======= VI. Evaluate the portfolio =======
        self.portfolio_evaluation()

        return self.portfolio

    # =========================================== Portfolio Management Methods =========================================== #
    def update_portfolio(
        self,
        Zup_exit_threshold: float = 3,
        leverage: bool = True,
        cash_margin: float = 0.2,
        kf_smooth_coefficient: float = 0.8,
        use_kf_weight: bool = True,
        risk_free_rate: float = 0.00035,
        collateralization_level: float = 1.25,
        haircut: float = 0.3,
    ):
        """
        Update the portfolio's informations when the date changes.

        Args:
            leverage (bool): If True, the portfolio will be leveraged.
            risk_free_rate (float): The risk-free rate used in the computation of the expected return.
            collateralisation_level (float): The collateralisation level required when taking a short position.
            haircut (float): The haircut applied to the cash needed to enter the position.

        Returns:
            new_portfolio (pd.DataFrame): DataFrame containing the updated portfolio's informations.
        """
        # ====== I. Initialize a new portfolio ======
        today = pd.to_datetime(self.date)
        train_start = today - pd.DateOffset(days=self.training_window)
        data = self.big_data.loc[train_start:today]
        new_portfolio = self.portfolio.copy()

        # ====== II. Compute the pairs information ======
        if not new_portfolio.empty:
            for index, row in new_portfolio.iterrows():
                # ------ 1. Get pair information -------
                assets = row["assets_name"]
                regression_weights = row["regression_weights"]
                normalized_weights = row["normalized_weights"]
                intercept = row["regression_weights"][0]
                combination_weights = row["combination_weights"]
                half_life = row["half_life"]
                sigma = row["volatility"]
                mu = 0
                theta = np.log(2) / half_life

                if combination_weights > 0:
                    # ------ 2. Compute the residuals -------
                    log_df = np.log(data[assets])
                    residuals = np.dot(log_df, regression_weights[1:]) + intercept
                    residuals_series = pd.Series(residuals, index=data.index)

                    # ------ 3. Compute the new information -------
                    adf_results = adfuller(residuals)
                    kpss_results = kpss(residuals, regression="c")

                    kalman_filter = Information.OU_KalmanFilter(
                        mean=mu,
                        theta=theta,
                        obs_sigma=sigma,
                        smooth_coefficient=kf_smooth_coefficient,
                    )
                    filtered_states, _ = kalman_filter.series_filter(
                        residuals=residuals_series
                    )

                    last_date = filtered_states.index[-1]
                    last_kf_value = filtered_states[-1]

                    kf_weight_numerator = last_kf_value - intercept
                    for i in range(1, len(regression_weights) - 1):
                        kf_weight_numerator -= (
                            regression_weights[i] * log_df[assets[i - 1]].loc[last_date]
                        )
                    kf_weight = kf_weight_numerator / log_df[assets[-1]].loc[last_date]

                    if use_kf_weight:
                        weights_copy = regression_weights.copy()
                        weights_copy[-1] = kf_weight
                        last_residual = last_kf_value

                        total_sum = sum(abs(x) for x in weights_copy[1:])
                        normalized_weights = []
                        for i in range(1, len(weights_copy)):  # Start from index 1
                            weight = weights_copy[i] / total_sum
                            normalized_weights.append(weight)
                    else:
                        last_residual = residuals_series.iloc[-1]

                    gap = abs(last_residual - mu)
                    z_score = (last_residual - mu) / sigma
                    leverage = bool(leverage)

                    expected_return, cash_needed, leverage_ratio = (
                        Information.get_expected_return(
                            normalized_weights=normalized_weights,
                            z_score=z_score,
                            half_life=half_life,
                            theta=theta,
                            sigma=sigma,
                            mu=mu,
                            gap=gap,
                            Zup_exit_threshold=Zup_exit_threshold,
                            leverage=leverage,
                            risk_free_rate=risk_free_rate,
                            cash_margin=cash_margin,
                            collateralization_level=collateralization_level,
                            haircut=haircut,
                        )
                    )

                    # ------ 4. Update the portfolio -------
                    new_portfolio.at[index, "normalized_weights"] = normalized_weights
                    new_portfolio.at[index, "kf_weight"] = kf_weight
                    new_portfolio.at[index, "z_score"] = z_score
                    new_portfolio.at[index, "gap"] = gap
                    new_portfolio.at[index, "adf_pvalue"] = adf_results[1]
                    new_portfolio.at[index, "kpss_pvalue"] = kpss_results[1]

                    new_portfolio.at[index, "expected_return"] = expected_return
                    new_portfolio.at[index, "cash_needed"] = cash_needed
                    new_portfolio.at[index, "leverage_ratio"] = leverage_ratio

        # ====== III. Update the portfolio ======
        self.portfolio = new_portfolio
        self.portfolio_evaluation()
        self.capital_engaged = self.cash_requirements()

        return new_portfolio

    # ---------------------------------------------------------------------------------------------------------------------#
    def check_exit_trade(
        self,
        Zup_exit_threshold: float = 3,
        Zlow_exit_threshold: float = 0,
        min_ret_exit_threshold: float = 0.00175,
    ):
        """
        Check if a pair should be exited from the portfolio.

        Args:
            Z_exit_threshold (float): The threshold for the z-score to exit a trade.
            min_ret_exit_threshold (float): The minimum expected return to exit a trade.

        Returns:
            new_portfolio (pd.DataFrame): DataFrame containing the updated portfolio's informations.
        """
        # ====== I. Initialize a new portfolio ======
        new_portfolio = self.portfolio.copy()

        # ====== II. Check the pairs to exit ======
        if not new_portfolio.empty:
            for index, row in new_portfolio.iterrows():
                # ------ 1. Get the pair information -------
                z_score = row["z_score"]
                expected_return = row["expected_return"]

                # ------ 2. Check the conditions over the price evolution -------
                if (
                    (abs(z_score) > Zup_exit_threshold)
                    or (abs(z_score) < Zlow_exit_threshold)
                    or (expected_return < min_ret_exit_threshold)
                ):
                    new_portfolio.at[index, "combination_weights"] = 0

                # ------ 3. Check the delisting condition -------
                assets = row["assets_name"]
                delisting_data = self.big_data[assets]
                delisting_date = delisting_data.index[
                    delisting_data.index <= self.delisting_check_date
                ][-1]
                if delisting_data.loc[delisting_date].isna().any():
                    new_portfolio.at[index, "combination_weights"] = 0

        # ====== III. Update the portfolio ======
        self.portfolio = new_portfolio
        self.portfolio_evaluation()
        self.capital_engaged = self.cash_requirements()

        return new_portfolio

    # ---------------------------------------------------------------------------------------------------------------------#
    def rebalance_portfolio(
        self,
        date: str,
        # Combination Informations parameters
        Zup_exit_threshold: float = 3,
        leverage: bool = True,
        cash_margin: float = 0.2,
        kf_smooth_coefficient: float = 0.8,
        use_kf_weight: bool = True,
        risk_free_rate: float = 0.00035,
        collateralization_level: float = 1.25,
        haircut: float = 0.3,
        # Portfolio optimization parameters
        model: str = "TargetVolatility",
        target_volatility: float = 0.2 / np.sqrt(252),
        min_weight: float = 0.03,
        upper_bound: float = 0.25,
        risk_measure: str = "MV",
        objective_function: str = "Sharpe",
        # Budget parameters
        budget: float = 1e6,
    ):
        # ======= 0. Update the date and initialize data =======
        self.date_update(date=date)
        self.combinations_list = []
        self.combinations_infos_list = []
        self.filtered_combinations_list = []

        # ======= I. Generate the combinations from the current portfolio =======
        for index, row in self.portfolio.iterrows():
            assets = row["assets_name"]
            weight = row["combination_weights"]
            if weight > 0:
                for group in self.data_storage:
                    if set(assets).issubset(group.columns):
                        combination_data = group[assets]
                        combination_data = combination_data.loc[
                            self.train_start_date : self.train_end_date
                        ]

                        # -------- 2. Remove the delisted stocks --------
                        delisting_date = group.index[
                            group.index <= self.delisting_check_date
                        ][-1]
                        delisted_stocks = group.columns[
                            group.loc[delisting_date].isna()
                        ]
                        delisted_stocks = delisted_stocks.intersection(
                            combination_data.columns
                        )

                        # -------- 3. Update the combinations list --------
                        if delisted_stocks.empty:
                            self.combinations_list.append(combination_data)

        # ======= II. Compute the combinations informations =======
        self.get_combination_informations(
            Zup_exit_threshold=Zup_exit_threshold,
            leverage=leverage,
            cash_margin=cash_margin,
            kf_smooth_coefficient=kf_smooth_coefficient,
            use_kf_weight=use_kf_weight,
            risk_free_rate=risk_free_rate,
            collateralization_level=collateralization_level,
            haircut=haircut,
        )

        # ======= III. We do not apply a filter on the combinations =======
        self.filtered_combinations_list = self.combinations_infos_list

        # ======= IV. Optimize the portfolio =======
        if len(self.filtered_combinations_list) > 0:
            self.portfolio_optimization(
                model=model,
                target_volatility=target_volatility,
                min_weight=min_weight,
                upper_bound=upper_bound,
                risk_measure=risk_measure,
                objective_function=objective_function,
            )
            self.round_weights(budget=budget)
        else:
            self.portfolio = pd.DataFrame()

        # ======= V. Evaluate the portfolio =======
        self.portfolio_evaluation()

        return self.portfolio


# ========================================================================================================== #
def daily_return(
    portfolio: pd.DataFrame,
    previous_portfolio: pd.DataFrame,
    big_data: pd.DataFrame,
    date: str,
    brokerages: float = 5.25e-4,
    slippage: float = 12e-4,
):
    """
    Compute the daily return of a portfolio of pairs between yesterday and today,
    accounting for brokerage fees and slippage.

    Args:
        portfolio (pd.DataFrame): contains the information about the current portfolio
        previous_portfolio (pd.DataFrame): contains the information about the previous day's portfolio
        big_data (pd.DataFrame): contains all prices data of the S&P500 over a large amount of time
        date (str): date at which we want to compute the portfolio return
        brokerages (float): percentage cost of executing trades
        slippage (float): additional cost due to slippage

    Returns:
        portfolio_return (float): the portfolio return after accounting for fees and slippage
    """
    # ====== I. Check conditions & get dates ======
    if portfolio.empty:
        return 0, 0, 0

    # ------ Get the date of today
    today = pd.to_datetime(date)
    if today not in big_data.index:
        return 0, 0, 0

    # ------ Find the most recent date available in the entire dataframe before 'today'
    recent_dates = big_data.index[big_data.index < today]
    most_recent_date = recent_dates[-1]

    # ====== II. Compute the portfolio return ======
    portfolio_return = 0
    assets_return = 0
    fees = 0
    for index, row in portfolio.iterrows():
        i = 0
        combination_return = 0
        for asset in row["assets_name"]:
            asset_weight = row["normalized_weights"][i]
            yesterday_price = big_data[asset].loc[most_recent_date]
            today_price = big_data[asset].loc[today]
            asset_return = (today_price / yesterday_price) - 1
            weighted_asset_return = asset_return * asset_weight
            combination_return += weighted_asset_return
            i += 1

        z_score = row["z_score"]
        if z_score < 0:
            combination_return = combination_return
        else:
            combination_return = -combination_return

        # ------- 2. Contribution to the portfolio -------
        combination_weight = row["combination_weights"]
        contribution_to_portfolio = combination_return * combination_weight

        assets_return += contribution_to_portfolio

        # ------- 3. Fees and slippage -------
        # i. Get previous weights
        assets = row["assets_name"]
        if previous_portfolio.empty:
            previous_position = pd.DataFrame()
        else:
            previous_position = previous_portfolio.loc[
                previous_portfolio["assets_name"].apply(lambda x: x == assets),
            ]

        new_assets_weights = np.array(row["normalized_weights"])
        new_weighted_assets_weights = new_assets_weights * combination_weight

        if not previous_position.empty:
            previous_combination_weights = previous_position[
                "combination_weights"
            ].iloc[0]
            previous_assets_weights = previous_position["normalized_weights"].iloc[0]
            previous_weighted_assets_weights = (
                np.array(previous_assets_weights) * previous_combination_weights
            )

            turnover = np.sum(
                np.abs(new_weighted_assets_weights - previous_weighted_assets_weights)
            )
        else:
            turnover = np.sum(np.abs(new_weighted_assets_weights))

        brokerage_fees = brokerages * turnover
        slippage_fees = slippage * turnover
        fees += brokerage_fees + slippage_fees

        contribution_to_portfolio -= brokerage_fees + slippage_fees

        portfolio_return += contribution_to_portfolio

    return portfolio_return, assets_return, fees


# ---------------------------------------------------------------------------------------------------------- #
def compute_stats(
    returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    frequence: str = "daily",
):
    """
    Compute the statistics of the investment.

    Args:
        returns (pd.Series): Series of returns of the investment.
        market_returns (pd.Series): Series of returns of the market index for comparison.
        risk_free_rate (float): Risk-free rate for certain calculations.
        frequence (str): Frequence of the returns.

    Returns:
        stats (dict): Dictionary containing the statistics of the investment, including:

        ======= Returns distribution statistics =======
        - **Expected Return**: The annualized mean return, indicating average performance.
        - **Volatility**: Standard deviation of returns, representing total risk.
        - **Downside Deviation**: Standard deviation of negative returns, used in risk-adjusted metrics like Sortino Ratio.
        - **Median Return**: The median of returns, a measure of central tendency.
        - **Skew** and **Kurtosis**: Describe the distribution shape, with skew indicating asymmetry and kurtosis indicating tail heaviness.

        ======= Risk measures =======
        - **Maximum Drawdown**: Largest observed loss from peak to trough, a measure of downside risk.
        - **Max Drawdown Duration**: Longest period to recover from drawdown, indicating risk recovery time.
        - **VaR 95** and **CVaR 95**: Value at Risk and Conditional Value at Risk at 95%, giving the maximum and average expected losses in worst-case scenarios.

        ======= Market sensitivity measures =======
        - **Beta**: Sensitivity to market movements.
        - **Alpha**: Risk-adjusted return above the market return.
        - **Upside/Downside Capture Ratios**: Percent of market gains or losses captured by the investment.
        - **Tracking Error**: Volatility of return differences from the market.

        ======= Performance measures =======
        - **Sharpe**: Risk-adjusted returns per unit of volatility.
        - **Sortino Ratio**: Risk-adjusted return accounting only for downside volatility.
        - **Treynor Ratio**: Return per unit of systematic (market) risk.
        - **Information Ratio**: Excess return per unit of tracking error.

        - **Sterling Ratio**: Return per unit of average drawdown.
        - **Calmar Ratio**: Return per unit of maximum drawdown.
    """
    # ======= 0. Initialization =======
    frequence_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequence = frequence_dict[frequence]

    cumulative_returns = (1 + returns).cumprod()

    # ======= I. Returns distribution statistics =======
    expected_return = returns.mean() * adjusted_frequence
    volatility = returns.std() * np.sqrt(adjusted_frequence)
    downside_deviation = (
        returns[returns < 0].std() * np.sqrt(adjusted_frequence)
        if returns[returns < 0].sum() != 0
        else 0
    )
    median_return = returns.median() * adjusted_frequence
    skew = returns.skew()
    kurtosis = returns.kurtosis()

    # ======= II. Risk measures =======
    # ------ Maximum Drawdown and Duration
    running_max = cumulative_returns.cummax().replace(0, 1e-10)
    drawdown = (cumulative_returns / running_max) - 1
    drawdown_durations = (
        (drawdown < 0).astype(int).groupby((drawdown == 0).cumsum()).cumsum()
    )

    maximum_drawdown = drawdown.min()
    max_drawdown_duration = drawdown_durations.max()

    # ------ Value at Risk and Conditional Value at Risk
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()

    # ======= III. Market sensitivity measures =======
    # ------ Beta and Alpha (Jensens's)
    beta = returns.cov(market_returns) / market_returns.var()
    alpha = expected_return - beta * (market_returns.mean() * adjusted_frequence)

    # ------ Capture Ratios
    upside_capture = (
        returns[market_returns > 0].mean() / market_returns[market_returns > 0].mean()
    )
    downside_capture = (
        returns[market_returns < 0].mean() / market_returns[market_returns < 0].mean()
    )

    # ------ Tracking Error
    tracking_error = returns.sub(market_returns).std() * np.sqrt(adjusted_frequence)

    # ======= IV. Performance measures =======
    # ------ Sharpe, Sortino, Treynor, and Information Ratios
    sharpe_ratio = (
        (expected_return - risk_free_rate) / volatility if volatility != 0 else 0
    )
    sortino_ratio = (
        expected_return / downside_deviation if downside_deviation != 0 else 0
    )
    treynor_ratio = expected_return / beta if beta != 0 else 0
    information_ratio = (
        (expected_return - market_returns.mean() * adjusted_frequence) / tracking_error
        if tracking_error != 0
        else 0
    )

    # ------ Sterling, and Calmar Ratios
    average_drawdown = (
        abs(drawdown[drawdown < 0].mean()) if drawdown[drawdown < 0].sum() != 0 else 0
    )
    sterling_ratio = (
        (expected_return - risk_free_rate) / average_drawdown
        if average_drawdown != 0
        else 0
    )
    calmar_ratio = (
        expected_return / abs(maximum_drawdown) if maximum_drawdown != 0 else 0
    )

    # ======= IV. Store the statistics =======
    stats = {
        "expected_return": expected_return,
        "volatility": volatility,
        "downside_deviation": downside_deviation,
        "median_return": median_return,
        "skew": skew,
        "kurtosis": kurtosis,
        "maximum_drawdown": maximum_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "beta": beta,
        "alpha": alpha,
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
        "tracking_error": tracking_error,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "treynor_ratio": treynor_ratio,
        "information_ratio": information_ratio,
        "sterling_ratio": sterling_ratio,
        "calmar_ratio": calmar_ratio,
    }

    return stats


# ========================================================================================================== #
def run_simulation(
    # Data
    data_storage: list,
    big_data: pd.DataFrame,
    risk_free_data: pd.DataFrame,
    market_data: pd.DataFrame,
    # Simulation Time Frame
    start_date: str,
    end_date: str,
    collateralization_level: float = 1.25,
    haircut: float = 0.3,
    budget: float = 1e6,
    # ------- Hyperparameters -------
    # Portfolio Management
    training_window: int = 365,
    min_sharpe_to_new_portfolio: float = 5,
    min_sharpe_to_trade: float = 10,
    min_sharpe_to_rebalance: float = 8,
    min_sharpe_spread_to_rebalance: float = 2,
    min_nb_pairs_to_new_portfolio: int = 3,
    min_nb_pairs_to_trade: int = 5,
    # Clustering parameters
    clustering_method: str = "riskfolio",
    linkage: str = "ward",
    n_clusters: int = 5,
    # Combinations generation parameters
    assets_per_comb: int = 2,
    max_shared_assets: int = 6,
    # Combination Informations parameters
    leverage: bool = True,
    cash_margin: float = 0.2,
    kf_smooth_coefficient: float = 0.8,
    use_kf_weight: bool = True,
    risk_free_rate: float = 0.00035,
    # Filter parameters
    adf_pvalue_threshold: float = 0.05,
    kpss_pvalue_threshold: float = 0.05,
    min_ret: float = 0.00175,
    lower_Z_bound: float = 1,
    upper_Z_bound: float = 2,
    # Portfolio optimization parameters
    model: str = "TargetVolatility",
    target_volatility: float = 0.2 / np.sqrt(252),
    min_weight: float = 0.03,
    upper_bound: float = 0.25,
    risk_measure: str = "MV",
    objective_function: str = "Sharpe",
    # Exit Strategy
    Zup_exit_threshold: float = 3,
    Zlow_exit_threshold: float = 0,
    min_ret_exit_threshold: float = 0.00175,
):
    # ======= 0. Initialize the date range =======
    time_start = time()
    nb_operations = 1

    date_range = big_data.index[big_data.index >= start_date]
    date_range = date_range[date_range <= end_date]
    first_day = date_range[0]

    # ======= I. Initialization =======
    # ------- 1. Initialize the history -------
    history = pd.DataFrame(
        index=date_range,
        columns=[
            "raw_returns",
            "assets_returns",
            "fees",
            "capital_engaged",
            "nb_pairs",
            "pf_expected_return",
            "pf_volatility",
            "pf_sharpe",
            "pf_half_life",
        ],
    )

    portfolio_history = {}

    # ------- 2. Initialize the portfolio -------
    previous_portfolio = pd.DataFrame()

    statsArb = StatsArb_Portfolio(
        data_storage=data_storage,
        big_data=big_data,
        date=first_day,
        training_window=training_window,
    )

    portfolio = statsArb.generate_portfolio(
        date=first_day,
        # Clustering parameters
        clustering_method=clustering_method,
        linkage=linkage,
        n_clusters=n_clusters,
        # Combinations generation parameters
        assets_per_comb=assets_per_comb,
        max_shared_assets=max_shared_assets,
        # Combination Informations parameters
        Zup_exit_threshold=Zup_exit_threshold,
        leverage=leverage,
        cash_margin=cash_margin,
        kf_smooth_coefficient=kf_smooth_coefficient,
        use_kf_weight=use_kf_weight,
        risk_free_rate=risk_free_rate,
        collateralization_level=collateralization_level,
        haircut=haircut,
        # Filter parameters
        adf_pvalue_threshold=adf_pvalue_threshold,
        kpss_pvalue_threshold=kpss_pvalue_threshold,
        min_ret=min_ret,
        lower_Z_bound=lower_Z_bound,
        upper_Z_bound=upper_Z_bound,
        # Portfolio optimization parameters
        model=model,
        target_volatility=target_volatility,
        min_weight=min_weight,
        upper_bound=upper_bound,
        risk_measure=risk_measure,
        objective_function=objective_function,
        # Budget parameters
        budget=budget,
    )

    capital_engaged = statsArb.capital_engaged
    nb_pairs = statsArb.nb_pairs
    pf_expected_return = statsArb.pf_expected_return
    pf_volatility = statsArb.pf_volatility
    pf_sharpe = statsArb.pf_sharpe
    pf_half_life = statsArb.pf_half_life

    history.loc[first_day] = {
        "raw_returns": 0,
        "assets_returns": 0,
        "fees": 0,
        "capital_engaged": capital_engaged,
        "nb_pairs": nb_pairs,
        "pf_expected_return": pf_expected_return,
        "pf_volatility": pf_volatility,
        "pf_sharpe": pf_sharpe,
        "pf_half_life": pf_half_life,
    }

    portfolio_history[first_day] = portfolio

    # ======= II. Loop over the dates =======
    for date in date_range[1:]:
        # ------- 1. Compute the daily return -------
        portfolio_return, assets_return, fees = daily_return(
            portfolio=portfolio,
            previous_portfolio=previous_portfolio,
            big_data=big_data,
            date=date,
            brokerages=9e-4,
            slippage=0,
        )

        # ------- 2. Save the portfolio before modifying it -------
        previous_portfolio = portfolio.copy()
        statsArb.date_update(date=date)

        # ------- 3. Update the current portfolio -------
        portfolio = statsArb.update_portfolio(
            Zup_exit_threshold=Zup_exit_threshold,
            leverage=leverage,
            cash_margin=cash_margin,
            kf_smooth_coefficient=kf_smooth_coefficient,
            use_kf_weight=use_kf_weight,
            risk_free_rate=risk_free_rate,
            collateralization_level=collateralization_level,
            haircut=haircut,
        )

        # ------- 4. Check the exit trades -------
        portfolio = statsArb.check_exit_trade(
            Zup_exit_threshold=Zup_exit_threshold,
            Zlow_exit_threshold=Zlow_exit_threshold,
            min_ret_exit_threshold=min_ret_exit_threshold,
        )

        # ------- 5. Evaluate the portfolio -------
        capital_engaged = statsArb.capital_engaged
        nb_pairs = statsArb.nb_pairs
        pf_expected_return = statsArb.pf_expected_return
        pf_volatility = statsArb.pf_volatility
        pf_sharpe = statsArb.pf_sharpe
        pf_half_life = statsArb.pf_half_life

        # ------ 6. Rebalance the portfolio if necessary -------
        if (pf_sharpe < min_sharpe_to_rebalance) and (not portfolio.empty):
            statsArb.rebalance_portfolio(
                date=date,
                leverage=leverage,
                cash_margin=cash_margin,
                kf_smooth_coefficient=kf_smooth_coefficient,
                use_kf_weight=use_kf_weight,
                risk_free_rate=risk_free_rate,
                collateralization_level=collateralization_level,
                haircut=haircut,
                model=model,
                target_volatility=target_volatility,
                min_weight=min_weight,
                upper_bound=upper_bound,
                risk_measure=risk_measure,
                objective_function=objective_function,
                budget=budget,
            )

            pf_sharpe_test = statsArb.pf_sharpe
            pf_sharpe_spread = pf_sharpe_test - pf_sharpe

            if pf_sharpe_spread > min_sharpe_spread_to_rebalance:
                print(f"----- Rebalancing the portfolio on {date}")
                nb_operations += 1

                portfolio = statsArb.portfolio
                capital_engaged = statsArb.capital_engaged
                nb_pairs = statsArb.nb_pairs
                pf_expected_return = statsArb.pf_expected_return
                pf_volatility = statsArb.pf_volatility
                pf_sharpe = statsArb.pf_sharpe
                pf_half_life = statsArb.pf_half_life
            else:
                statsArb.portfolio = portfolio
                statsArb.portfolio_evaluation()

        # ------- 6. Check if the conditions of computing a new portfolio are met -------
        if (pf_sharpe < min_sharpe_to_new_portfolio) or (
            nb_pairs < min_nb_pairs_to_new_portfolio
        ):
            print("------------------")
            portfolio = statsArb.generate_portfolio(
                date=date,
                clustering_method=clustering_method,
                linkage=linkage,
                n_clusters=n_clusters,
                assets_per_comb=assets_per_comb,
                max_shared_assets=max_shared_assets,
                Zup_exit_threshold=Zup_exit_threshold,
                leverage=leverage,
                cash_margin=cash_margin,
                kf_smooth_coefficient=kf_smooth_coefficient,
                use_kf_weight=use_kf_weight,
                risk_free_rate=risk_free_rate,
                collateralization_level=collateralization_level,
                haircut=haircut,
                adf_pvalue_threshold=adf_pvalue_threshold,
                kpss_pvalue_threshold=kpss_pvalue_threshold,
                min_ret=min_ret,
                lower_Z_bound=lower_Z_bound,
                upper_Z_bound=upper_Z_bound,
                model=model,
                target_volatility=target_volatility,
                min_weight=min_weight,
                upper_bound=upper_bound,
                risk_measure=risk_measure,
                objective_function=objective_function,
                budget=budget,
            )

            capital_engaged = statsArb.capital_engaged
            nb_pairs = statsArb.nb_pairs
            pf_expected_return = statsArb.pf_expected_return
            pf_volatility = statsArb.pf_volatility
            pf_sharpe = statsArb.pf_sharpe
            pf_half_life = statsArb.pf_half_life

            # ------- 7. Check if the current portfolio meet our conditions -------
            if (pf_sharpe < min_sharpe_to_trade) or (nb_pairs < min_nb_pairs_to_trade):
                statsArb.portfolio = pd.DataFrame()
                statsArb.portfolio_evaluation()
                portfolio = statsArb.portfolio

                capital_engaged = statsArb.capital_engaged
                nb_pairs = statsArb.nb_pairs
                pf_expected_return = statsArb.pf_expected_return
                pf_volatility = statsArb.pf_volatility
                pf_sharpe = statsArb.pf_sharpe
                pf_half_life = statsArb.pf_half_life
            else:
                print(f"New Portfolio generated on {date}")
                nb_operations += 1

        # ------- 8. Store the results -------
        history.loc[date] = {
            "raw_returns": portfolio_return,
            "assets_returns": assets_return,
            "fees": fees,
            "capital_engaged": capital_engaged,
            "nb_pairs": nb_pairs,
            "pf_expected_return": pf_expected_return,
            "pf_volatility": pf_volatility,
            "pf_sharpe": pf_sharpe,
            "pf_half_life": pf_half_life,
        }

        portfolio_history[date] = portfolio

    # # ======= III. Compute the statistics =======
    history = history.dropna()
    history["raw_returns"] = history["raw_returns"].astype(float)
    history["assets_returns"] = history["assets_returns"].astype(float)
    history["fees"] = history["fees"].astype(float)
    history["capital_engaged"] = history["capital_engaged"].astype(float)
    history["nb_pairs"] = history["nb_pairs"].astype(int)
    history["pf_expected_return"] = history["pf_expected_return"].astype(float)
    history["pf_volatility"] = history["pf_volatility"].astype(float)
    history["pf_sharpe"] = history["pf_sharpe"].astype(float)
    history["pf_half_life"] = history["pf_half_life"].astype(float)

    date_range = history.index
    history["rf_returns"] = risk_free_data.reindex(history.index, fill_value=0)
    history["market_returns"] = market_data.reindex(history.index, fill_value=0)
    # history["rf_returns"] = risk_free_data.loc[date_range].fillna(0)
    # history["market_returns"] = market_data.loc[date_range].fillna(0)

    history["boosted_returns"] = history["raw_returns"] + history["rf_returns"] * (
        1 - history["capital_engaged"]
    )
    history["boosted_returns"] = history["boosted_returns"].astype(float)

    performance_metrics = compute_stats(
        returns=history["raw_returns"], market_returns=history["market_returns"]
    )

    time_end = time()
    simulation_time = time_end - time_start

    maturity = (date_range[-1] - date_range[0]).days
    maturity = max(1, round(maturity / 365))
    trading_score = Information.trading_score(
        nb_operations=nb_operations,
        min_nb_operations=3,
        target_nb_operations=30,
        maturity=maturity,
    )

    # ======= IV. Store the results  =======
    hyperparameters = {
        "training_window": training_window,
        "min_sharpe_to_new_portfolio": min_sharpe_to_new_portfolio,
        "min_sharpe_to_trade": min_sharpe_to_trade,
        "min_sharpe_to_rebalance": min_sharpe_to_rebalance,
        "min_sharpe_spread_to_rebalance": min_sharpe_spread_to_rebalance,
        "min_nb_pairs_to_new_portfolio": min_nb_pairs_to_new_portfolio,
        "min_nb_pairs_to_trade": min_nb_pairs_to_trade,
        "clustering_method": clustering_method,
        "linkage": linkage,
        "n_clusters": n_clusters,
        "assets_per_comb": assets_per_comb,
        "max_shared_assets": max_shared_assets,
        "leverage": leverage,
        "cash_margin": cash_margin,
        "kf_smooth_coefficient": kf_smooth_coefficient,
        "use_kf_weight": use_kf_weight,
        "risk_free_rate": risk_free_rate,
        "adf_pvalue_threshold": adf_pvalue_threshold,
        "kpss_pvalue_threshold": kpss_pvalue_threshold,
        "min_ret": min_ret,
        "lower_Z_bound": lower_Z_bound,
        "upper_Z_bound": upper_Z_bound,
        "model": model,
        "target_volatility": target_volatility,
        "min_weight": min_weight,
        "upper_bound": upper_bound,
        "risk_measure": risk_measure,
        "objective_function": objective_function,
        "Zup_exit_threshold": Zup_exit_threshold,
        "Zlow_exit_threshold": Zlow_exit_threshold,
        "min_ret_exit_threshold": min_ret_exit_threshold,
    }

    simulation_results = {
        "history": history,
        "portfolio_history": portfolio_history,
        "simulation_time": simulation_time,
        "performance_metrics": performance_metrics,
        "hyperparameters": hyperparameters,
        "trading_score": trading_score,
    }

    return simulation_results
