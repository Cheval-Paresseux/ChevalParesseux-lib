"""
# Description: Different ways of optimizing the hyperparameters of a strategy using a genetic algorithm or a search algorithm.
_____
GeneticAlgorithm: Class that runs the genetic algorithm to optimize the hyperparameters of the strategy.
MilleFeuille: Class that runs a search to find the best hyperparameters for each part of the strategy (portfolio management, exit strategy, filter, Kalman Smooth, clustering).
_____
POTENTIAL IMPROVEMENTS:
    - Improve the Grid Search algorithm to be more efficient.
    - Apply a more advanced optimization algorithm.
    - Apply a learning algorithm to optimize and adjust the hyperparameters over time.
    - Optimize the computation time.
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from IPython.display import clear_output

import os
import sys

sys.path.append(os.path.abspath(".."))
import models.Simulation as sim

import warnings

warnings.filterwarnings("ignore")


# ========================================================================================================== #
class GeneticAlgorithm:
    def __init__(
        self,
        data_storage: list,
        big_data: pd.DataFrame,
        risk_free_data: pd.DataFrame,
        market_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        n_jobs: int = 2,
        collateralization_level: float = 1.25,
        haircut: float = 0.3,
        budget: float = 1e6,
    ):
        """
        Initialize the class with the input data and parameters.

        Args:
            data_storage (list): List of pd.DataFrame with historical data for each asset.
            big_data (pd.DataFrame): DataFrame with historical data for all assets.
            risk_free_data (pd.DataFrame): DataFrame with historical data for the risk-free asset.
            market_data (pd.DataFrame): DataFrame with historical data for the market index.
            start_date (str): Start date for the simulation.
            end_date (str): End date for the simulation.
            n_jobs (int): Number of parallel jobs to run.
            collateralisation_level (float): Collateralisation level required when shorting a stock.
            haircut (float): Haircut applied to the value of the assets when used as collateral.
            budget (float): Initial budget for the simulation.
        """
        # ======= I. Input data =======
        self.data_storage = data_storage
        self.big_data = big_data
        self.risk_free_data = risk_free_data
        self.market_data = market_data
        self.start_date = start_date
        self.end_date = end_date
        self.n_jobs = n_jobs
        self.collateralization_level = collateralization_level
        self.haircut = haircut
        self.budget = budget

        # ======= II. Pre-stored data =======
        self.available_params = {
            # Data
            "data_storage": [self.data_storage],
            "big_data": [self.big_data],
            "risk_free_data": [self.risk_free_data],
            "market_data": [self.market_data],
            # Simulation Time Frame
            "start_date": [self.start_date],
            "end_date": [self.end_date],
            "collateralization_level": [self.collateralization_level],
            "haircut": [self.haircut],
            "budget": [self.budget],
            # ------- Hyperparameters -------
            # Portfolio Management
            "training_window": [180, 200, 220, 250, 280, 300, 330, 365, 400, 450, 500, 547],
            "min_sharpe_to_new_portfolio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2],
            "min_sharpe_to_trade": [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1, 1.1, 1.2, 1.3, 1.4, 1.5],
            "min_sharpe_to_rebalance": [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2],
            "min_sharpe_spread_to_rebalance": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            "min_nb_pairs_to_new_portfolio": [0, 1, 2, 3],
            "min_nb_pairs_to_trade": [0, 1, 2, 3],
            # Clustering parameters
            "clustering_method": ["riskfolio", "dtw"],
            "linkage": ["ward", "DBHT"],
            "n_clusters": [5, 8, 10, 13, 16, 20],
            # Combinations generation parameters
            "assets_per_comb": [2],
            "max_shared_assets": [3, 4, 5, 6],
            # Combination Informations parameters
            "leverage": [False],
            "cash_margin": [0.2],
            "kf_smooth_coefficient": [0.5, 0.6, 0.7, 0.8, 0.9],
            "use_kf_weight": [True, False],
            "risk_free_rate": [0.00035],
            # Filter parameters
            "adf_pvalue_threshold": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            "kpss_pvalue_threshold": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            "min_ret": [0, 0.001, 0.0015, 0.00175, 0.002, 0.0025],
            "lower_Z_bound": [0, 0.3, 0.5, 0.8, 1, 1.3, 1.5, 1.8, 2, 2.2, 2.5],
            "upper_Z_bound": [2, 2.2, 2.5, 2.8, 3, 3.2, 3.5, 3.8, 4, 4.2, 4.5, 4.8, 5],
            # Portfolio optimization parameters
            "model": ["TargetVolatility", "Classic", "EqualWeights", "InverseVolatility", "RiskParity"],
            "target_volatility": [
                0.2 / np.sqrt(252),
                0.25 / np.sqrt(252),
                0.3 / np.sqrt(252),
                0.35 / np.sqrt(252),
                0.4 / np.sqrt(252),
            ],
            "min_weight": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            "upper_bound": [0.15, 0.2, 0.22, 0.25, 0.28, 0.3, 0.33, 0.35, 0.38, 0.4, 0.43, 0.45, 0.48, 0.5],
            "risk_measure": ["MV", "CDaR"],
            "objective_function": ["Sharpe", "MinRisk"],
            # Exit Strategy
            "Zup_exit_threshold": [2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4],
            "Zlow_exit_threshold": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "min_ret_exit_threshold": [0, 0.001, 0.0015, 0.00175, 0.002, 0.0025],
        }

        # ======= II. Storing the results =======
        self.evolution_history = {}

    # ============================================= SIMULATION METHODS ============================================= #
    def run_simulation(self, params: dict = None):
        """
        Simulate the strategy with the given hyperparameters.

        Args:
            params (dict): Dictionary with the hyperparameters for the simulation.

        Returns:
            results (pd.DataFrame): DataFrame with the simulation results.
        """
        # ======= I. Run the simulation =======
        simulation_results = sim.run_simulation(**params)

        # ======= II. Store the results =======
        history = simulation_results["history"]
        hyperparameters = simulation_results["hyperparameters"]
        simulation_time = simulation_results["simulation_time"]
        performance_metrics = simulation_results["performance_metrics"]
        trading_score = simulation_results["trading_score"]

        # ======= III. Put the results as dataframe =======
        results = pd.DataFrame()
        results["history"] = [history]
        results["hyperparameters"] = [hyperparameters]
        results["simulation_time"] = simulation_time
        results["trading_score"] = trading_score

        metrics_df = pd.DataFrame([performance_metrics])
        results = pd.concat([results, metrics_df], axis=1)

        return results

    # ---------------------------------------------------------------------------------------------------------------#
    def parallel_simulations(self, params_list: list):
        """
        Simulate the strategy with multiple sets of hyperparameters in parallel.

        Args:
            params_list (list): List of dictionaries with the hyperparameters for each simulation.

        Returns:
            population_results (pd.DataFrame): DataFrame with the simulation results for all hyperparameters.
        """
        # ======= I. Run the simulations in parallel =======
        list_of_params_list = [params_list[i : i + self.n_jobs] for i in range(0, len(params_list), self.n_jobs)]
        list_of_sim_df = []

        for params_list in list_of_params_list:
            simulations_results = Parallel(n_jobs=self.n_jobs)(delayed(self.run_simulation)(params) for params in params_list)
            sim_df = pd.concat(simulations_results, axis=0, ignore_index=True)
            list_of_sim_df.append(sim_df)

        # ======= II. Store the results in a dataframe =======
        population_results = pd.concat(list_of_sim_df, axis=0, ignore_index=True)

        clear_output(wait=True)

        return population_results

    # ============================================= EVOLUTION METHODS ============================================= #
    def clear_params_overlap(self, params: dict):
        """
        Adjust the hyperparameters to avoid overlap between the thresholds.

        Args:
            params (dict): Dictionary with the hyperparameters for the simulation.

        Returns:
            params (dict): Dictionary with the adjusted hyperparameters.
        """
        # ======= I. Clear the overlap for entry/exit thresholds =======
        if params["lower_Z_bound"] >= params["upper_Z_bound"]:
            params["lower_Z_bound"] = params["upper_Z_bound"] - 1

        if params["Zup_exit_threshold"] < params["upper_Z_bound"]:
            params["Zup_exit_threshold"] = params["upper_Z_bound"]

        if params["Zlow_exit_threshold"] > params["lower_Z_bound"]:
            params["Zlow_exit_threshold"] = params["lower_Z_bound"]

        return params

    # ---------------------------------------------------------------------------------------------------------------#
    def get_random_params(self, nb_params: int):
        """
        Generate a list of random hyperparameters.

        Args:
            nb_params (int): Number of random hyperparameters to generate.

        Returns:
            params_list (list): List of dictionaries with the random hyperparameters.
        """
        params_list = []
        for _ in range(nb_params):
            random_params = {}
            for key, value in self.available_params.items():
                # Directly assign if single value; use random choice if multiple values
                if len(value) == 1:
                    random_params[key] = value[0]
                else:
                    chosen_value = np.random.choice(value)
                    random_params[key] = type(value[0])(chosen_value)

            random_params = self.clear_params_overlap(random_params)
            params_list.append(random_params)

        return params_list

    # ---------------------------------------------------------------------------------------------------------------#
    def filter_results(self, population_results: pd.DataFrame):
        """
        Apply a filter to the results to select the best performing strategies.

        Args:
            population_results (pd.DataFrame): DataFrame with the simulations results.
        """
        # ======= I. Initialize the new dataframe  =======
        relative_results = population_results.copy()
        important_columns = [
            "sharpe_ratio",
            "sortino_ratio",
            "treynor_ratio",
            "information_ratio",
            "sterling_ratio",
            "calmar_ratio",
        ]

        # ======= II. Normalize the results, column by column =======
        for column in important_columns:
            if column in relative_results.columns:
                min_value = relative_results[column].min()
                max_value = relative_results[column].max()

                if max_value > min_value:
                    relative_results[column] = (relative_results[column] - min_value) / (max_value - min_value)
                else:
                    relative_results[column] = 0.5

        # ======= III. Compute the average score of each simulation =======
        relative_results["total_score"] = relative_results[important_columns].mean(axis=1)
        relative_results["total_score"] = relative_results["total_score"] * relative_results["trading_score"]

        return relative_results

    # ---------------------------------------------------------------------------------------------------------------#
    def select_parents(self, relative_results: pd.DataFrame, num_parents: int):
        """
        Select the best performing strategies to be the parents of the next generation.

        Args:
            relative_results (pd.DataFrame): DataFrame with the simulations results.
            num_parents (int): Number of parents to select.

        Returns:
            parents (pd.DataFrame): DataFrame with the selected parents.
        """
        # ======= I. Sort the results by the total score =======
        relative_results_sorted = relative_results.sort_values(by="total_score", ascending=False)

        # ======= II. Select the best performing strategies as parents =======
        parents = relative_results_sorted.head(num_parents)

        return parents

    # ---------------------------------------------------------------------------------------------------------------#
    def crossover(self, parent1: dict, parent2: dict):
        """
        Apply crossover to two parents to create a new offspring.

        Args:
            parent1 (dict): Dictionary with the hyperparameters of the first parent.
            parent2 (dict): Dictionary with the hyperparameters of the second parent.

        Returns:
            offspring (dict): Dictionary with the hyperparameters of the offspring.
        """
        # ======= I. Randomly select a crossover point =======
        crossover_point = np.random.randint(1, len(parent1))

        # ======= II. Create the offspring =======
        offspring = {}
        for i, key in enumerate(parent1.keys()):
            offspring[key] = parent1[key] if i < crossover_point else parent2[key]

        return offspring

    # ---------------------------------------------------------------------------------------------------------------#
    def mutate(self, params: dict, mutation_rate: float):
        """
        Apply mutation to the hyperparameters of a strategy.

        Args:
            params (dict): Dictionary with the hyperparameters of the strategy.
            mutation_rate (float): Probability of mutation for each hyperparameter.

        Returns:
            params (dict): Dictionary with the mutated hyperparameters.
        """

        for key, _ in params.items():
            if np.random.rand() < mutation_rate:
                chosen_value = np.random.choice(self.available_params[key])
                params[key] = type(self.available_params[key][0])(chosen_value)

        return params

    # ---------------------------------------------------------------------------------------------------------------#
    def generate_new_population(self, parents_pool: pd.DataFrame, population_size: int, mutation_rate: float):
        """
        Generate a new population of hyperparameters using the parents pool.

        Args:
            parents_pool (pd.DataFrame): DataFrame with the selected parents.
            population_size (int): Number of hyperparameters to generate.
            mutation_rate (float): Probability of mutation for each hyperparameter.

        Returns:
            new_population_params (list): List of dictionaries with the new hyperparameters.
        """

        fixed_params = {
            "data_storage": self.data_storage,
            "big_data": self.big_data,
            "risk_free_data": self.risk_free_data,
            "market_data": self.market_data,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "collateralization_level": self.collateralization_level,
            "haircut": self.haircut,
            "budget": self.budget,
        }
        new_population_params = []

        while len(new_population_params) < population_size:
            # ------- Randomly select two parents -------
            parent1 = parents_pool.sample(1).iloc[0]["hyperparameters"]
            parent2 = parents_pool.sample(1).iloc[0]["hyperparameters"]

            # ------- Apply crossover and mutation -------
            offspring_hyperparams = self.crossover(parent1=parent1, parent2=parent2)
            offspring_hyperparams = self.mutate(params=offspring_hyperparams, mutation_rate=mutation_rate)

            offspring_hyperparams = {**fixed_params, **offspring_hyperparams}
            # ------- Append the offspring to the new population -------
            new_population_params.append(offspring_hyperparams)

        return new_population_params

    # ============================================= RUNNING METHOD ============================================= #
    def run_genetic_algorithm(self, nb_generations: int, population_size: int, mutation_rate: float, num_parents: int):
        """
        Run the genetic algorithm to optimize the hyperparameters of the strategy.

        Args:
            nb_generations (int): Number of generations to run.
            population_size (int): Number of hyperparameters to generate in each generation.
            mutation_rate (float): Probability of mutation for each hyperparameter.

        Returns:
            population_results (pd.DataFrame): DataFrame with the simulation results for all hyperparameters.
        """
        # ======= I. Initialize the first population =======
        population_params = self.get_random_params(nb_params=population_size)
        population_results = self.parallel_simulations(population_params)
        self.evolution_history["generation_0"] = population_results

        # ======= II. Run the genetic algorithm =======
        for i in tqdm(range(nb_generations)):
            # ------- 1. Filter the results of the current population -------
            relative_results = self.filter_results(population_results)
            relative_results = relative_results.dropna(axis=0)
            # ------- 2. Select the best performing strategies as parents -------
            parents_pool = self.select_parents(relative_results, num_parents=num_parents)

            # ------- 3. Generate a new population using the parents -------
            updated_mutation_rate = mutation_rate * (1 - i / nb_generations)
            new_population_params = self.generate_new_population(parents_pool, population_size, updated_mutation_rate)
            new_population_results = self.parallel_simulations(new_population_params)

            population_results = pd.concat([population_results, new_population_results], axis=0, ignore_index=True)

            # ------- 4. Store the results -------
            self.evolution_history["generation_{}".format(i + 1)] = new_population_results

        # ======= III. Return the final population results =======
        last_generation = new_population_results.dropna(axis=0)
        filtered_results = self.filter_results(last_generation)
        filtered_results = filtered_results.sort_values(by="total_score", ascending=False)

        return last_generation, filtered_results


class MilleFeuille:
    def __init__(
        self,
        data_storage: list,
        big_data: pd.DataFrame,
        risk_free_data: pd.DataFrame,
        market_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        n_jobs: int = 5,
        nb_paths_explored: int = 2,
        collateralization_level: float = 1.25,
        haircut: float = 0.3,
        budget: float = 1e6,
    ):
        """
        Initialize the class with the input data and parameters.

        Args:
            data_storage (list): List of pd.DataFrame with historical data for each asset.
            big_data (pd.DataFrame): DataFrame with historical data for all assets.
            risk_free_data (pd.DataFrame): DataFrame with historical data for the risk-free asset.
            market_data (pd.DataFrame): DataFrame with historical data for the market index.
            start_date (str): Start date for the simulation.
            end_date (str): End date for the simulation.
            n_jobs (int): Number of parallel jobs to run.
            collateralisation_level (float): Collateralisation level required when shorting a stock.
            haircut (float): Haircut applied to the value of the assets when used as collateral.
            budget (float): Initial budget for the simulation.
        """
        # ======= I. Input data =======
        self.n_jobs = n_jobs
        self.nb_paths_explored = nb_paths_explored

        self.fixed_params = {
            "data_storage": data_storage,
            "big_data": big_data,
            "risk_free_data": risk_free_data,
            "market_data": market_data,
            "start_date": start_date,
            "end_date": end_date,
            "collateralization_level": collateralization_level,
            "haircut": haircut,
            "budget": budget,
        }

        # ======= II. Initial parameters =======
        self.initial_params = {
            # Data
            "data_storage": data_storage,
            "big_data": big_data,
            "risk_free_data": risk_free_data,
            "market_data": market_data,
            # Simulation Time Frame
            "start_date": start_date,
            "end_date": end_date,
            "collateralization_level": collateralization_level,
            "haircut": haircut,
            "budget": budget,
            # ------- Hyperparameters -------
            # Portfolio Management
            "training_window": 365,
            "min_sharpe_to_new_portfolio": 0.8,
            "min_sharpe_to_trade": 1,
            "min_sharpe_to_rebalance": 1.1,
            "min_sharpe_spread_to_rebalance": 0,
            "min_nb_pairs_to_new_portfolio": 0,
            "min_nb_pairs_to_trade": 0,
            # Clustering parameters
            "clustering_method": "riskfolio",
            "linkage": "ward",
            "n_clusters": 10,
            # Combinations generation parameters
            "assets_per_comb": 2,
            "max_shared_assets": 6,
            # Combination Informations parameters
            "leverage": False,
            "cash_margin": 0.2,
            "kf_smooth_coefficient": 0.7,
            "use_kf_weight": True,
            "risk_free_rate": 0.00035,
            # Filter parameters
            "adf_pvalue_threshold": 0.05,
            "kpss_pvalue_threshold": 0.05,
            "min_ret": 0.001,
            "lower_Z_bound": 1,
            "upper_Z_bound": 3,
            # Portfolio optimization parameters
            "model": "Classic",
            "target_volatility": 0.2 / np.sqrt(252),
            "min_weight": 0.03,
            "upper_bound": 0.3,
            "risk_measure": "MV",
            "objective_function": "Sharpe",
            # Exit Strategy
            "Zup_exit_threshold": 3.5,
            "Zlow_exit_threshold": 0,
            "min_ret_exit_threshold": 0,
        }

        # ======= III. Storing the results =======
        self.evolution_history = {}
        self.current_top_params = [self.initial_params]

        self.best_portfolio_params = None
        self.best_exit_params = None
        self.best_filter_params = None
        self.best_kalmansmooth_params = None
        self.best_clustering_params = None
        self.best_management_params = None

    # ============================================= SIMULATION METHODS ============================================= #
    def run_simulation(self, params: dict = None):
        """
        Simulate the strategy with the given hyperparameters.

        Args:
            params (dict): Dictionary with the hyperparameters for the simulation.

        Returns:
            results (pd.DataFrame): DataFrame with the simulation results.
        """
        # ======= I. Run the simulation =======
        simulation_results = sim.run_simulation(**params)

        # ======= II. Store the results =======
        hyperparameters = simulation_results["hyperparameters"]
        simulation_time = simulation_results["simulation_time"]
        performance_metrics = simulation_results["performance_metrics"]
        trading_score = simulation_results["trading_score"]

        # ======= III. Put the results as dataframe =======
        results = pd.DataFrame()
        results["hyperparameters"] = [hyperparameters]
        results["simulation_time"] = simulation_time
        results["trading_score"] = trading_score

        metrics_df = pd.DataFrame([performance_metrics])
        results = pd.concat([results, metrics_df], axis=1)

        return results

    # ---------------------------------------------------------------------------------------------------------------#
    def parallel_simulations(self, params_list: list):
        """
        Simulate the strategy with multiple sets of hyperparameters in parallel.

        Args:
            params_list (list): List of dictionaries with the hyperparameters for each simulation.

        Returns:
            population_results (pd.DataFrame): DataFrame with the simulation results for all hyperparameters.
        """
        # ======= I. Run the simulations in parallel =======
        list_of_params_list = [params_list[i : i + self.n_jobs] for i in range(0, len(params_list), self.n_jobs)]
        list_of_sim_df = []

        for params_sublist in tqdm(list_of_params_list):
            simulations_results = Parallel(n_jobs=self.n_jobs)(delayed(self.run_simulation)(params) for params in params_sublist)
            sim_df = pd.concat(simulations_results, axis=0, ignore_index=True)
            list_of_sim_df.append(sim_df)

        # ======= II. Store the results in a dataframe =======
        population_results = pd.concat(list_of_sim_df, axis=0, ignore_index=True)

        clear_output(wait=True)

        return population_results

    # ============================================= SEARCHING METHODS ============================================= #
    def portfolio_search(self):
        """
        This method runs a search to find the best portfolio management method for the strategy.
        Total number of simulations: 11

        Returns:
            search_results (pd.DataFrame): DataFrame with the simulation results for all portfolio management methods.
            best_portfolio_method (dict): Dictionary with the hyperparameters of the best portfolio management method.
        """
        # ======= I. Define the portfolio methods to test =======
        non_params_models = ["InverseVolatility", "RiskParity", "EqualWeights"]
        risk_measures = ["MV", "CDaR"]
        target_volatility = [0.2 / np.sqrt(252), 0.25 / np.sqrt(252), 0.3 / np.sqrt(252), 0.35 / np.sqrt(252), 0.4 / np.sqrt(252)]
        objective_functions = ["Sharpe", "MinRisk"]
        params_list = self.current_top_params.copy()

        # ------- Non parametrics Portfolio methods : X*3 -------
        for current_top_params in self.current_top_params:
            for method in non_params_models:
                params = current_top_params.copy()
                params["model"] = method
                params_list.append(params)

        # ------- Markowitz Portfolio methods : X*3 -------
        for current_top_params in self.current_top_params:
            for risk_measure in risk_measures:
                for objective_function in objective_functions:
                    if risk_measure == "MV" and objective_function == "Sharpe":  # Classic/MV/Sharpe is the default value, it is tested in the initial params
                        continue
                    params = current_top_params.copy()
                    params["model"] = "Classic"
                    params["risk_measure"] = risk_measure
                    params["objective_function"] = objective_function
                    params_list.append(params)

        # ------- Target Volatility Portfolio methods : X*5 -------
        for current_top_params in self.current_top_params:
            for target_vol in target_volatility:
                params = current_top_params.copy()
                params["model"] = "TargetVolatility"
                params["target_volatility"] = target_vol
                params_list.append(params)

        # ======= II. Run the simulations =======
        search_results = self.parallel_simulations(params_list)
        sorted_results = search_results.sort_values(by="sharpe_ratio", ascending=False)

        best_portfolio_params = []
        for i in range(self.nb_paths_explored):
            hyper_params = sorted_results.iloc[i]["hyperparameters"]
            params = {**self.fixed_params, **hyper_params}
            best_portfolio_params.append(params)

        # ======= III. Save the results =======
        self.evolution_history["portfolio_search"] = search_results
        self.best_portfolio_params = best_portfolio_params
        self.current_top_params = best_portfolio_params

        print("Portfolio model search completed.")

        return sorted_results, best_portfolio_params

    # ---------------------------------------------------------------------------------------------------------------#
    def exit_search(self):
        # ======= I. Define the exit methods to test =======
        Zup_exit_threshold = [2.5, 2.75, 3, 3.25, 3.75, 4]  # 3.5 is the default value, it is tested in the initial params
        Zlow_exit_threshold = [0.25, 0.5, 0.75, 1]  # 0 is the default value, it is tested in the initial params
        min_ret_exit_threshold = [0.001, 0.0015, 0.002]  # 0 is the default value, it is tested in the initial params
        params_list = self.current_top_params.copy()

        # ------- Upper Exit methods : X*7 -------
        for current_top_params in self.current_top_params:
            for Zup in Zup_exit_threshold:
                params = current_top_params.copy()
                params["Zup_exit_threshold"] = Zup
                params_list.append(params)

        # ------- Lower Exit methods : X*4 -------
        for current_top_params in self.current_top_params:
            for Zlow in Zlow_exit_threshold:
                params = current_top_params.copy()
                params["Zlow_exit_threshold"] = Zlow
                params_list.append(params)

        # ------- Return Exit methods : X*3 -------
        for current_top_params in self.current_top_params:
            for min_ret in min_ret_exit_threshold:
                params = current_top_params.copy()
                params["min_ret_exit_threshold"] = min_ret
                params_list.append(params)

        # ======= II. Run the simulations =======
        search_results = self.parallel_simulations(params_list)
        sorted_results = search_results.sort_values(by="sharpe_ratio", ascending=False)

        best_exit_params = []
        for i in range(self.nb_paths_explored):
            hyper_params = sorted_results.iloc[i]["hyperparameters"]
            params = {**self.fixed_params, **hyper_params}
            best_exit_params.append(params)

        # ======= III. Save the results =======
        self.evolution_history["exit_search"] = search_results
        self.best_exit_params = best_exit_params
        self.current_top_params = best_exit_params

        print("Exit strategy search completed.")

        return sorted_results, best_exit_params

    # ---------------------------------------------------------------------------------------------------------------#
    def filter_search(self):
        # ======= I. Define the filter methods to test =======
        adf_pvalue_threshold = [0.01, 0.03, 0.07, 0.1]  # 0.05 is the default value, it is tested in the initial params
        kpss_pvalue_threshold = [0.01, 0.03, 0.07, 0.1]  # 0.05 is the default value, it is tested in the initial params
        min_ret = [0, 0.0015, 0.002, 0.0025]  # 0.001 is the default value, it is tested in the initial params

        params_list = self.current_top_params.copy()

        # ------- ADF p-value threshold : X*4 -------
        for current_top_params in self.current_top_params:
            for adf in adf_pvalue_threshold:
                params = current_top_params.copy()
                params["adf_pvalue_threshold"] = adf
                params_list.append(params)

        # ------- KPSS p-value threshold : X*4 -------
        for current_top_params in self.current_top_params:
            for kpss in kpss_pvalue_threshold:
                params = current_top_params.copy()
                params["kpss_pvalue_threshold"] = kpss
                params_list.append(params)

        # ------- Minimum return threshold : X*4 -------
        for current_top_params in self.current_top_params:
            for ret in min_ret:
                params = current_top_params.copy()
                params["min_ret"] = ret
                params_list.append(params)

        # ======= II. Run the simulations =======
        search_results = self.parallel_simulations(params_list)
        sorted_results = search_results.sort_values(by="sharpe_ratio", ascending=False)

        best_filter_params = []
        for i in range(self.nb_paths_explored):
            hyper_params = sorted_results.iloc[i]["hyperparameters"]
            params = {**self.fixed_params, **hyper_params}
            best_filter_params.append(params)

        # ======= III. Save the results =======
        self.evolution_history["filter_search"] = search_results
        self.best_filter_params = best_filter_params
        self.current_top_params = best_filter_params

        print("Filter search completed.")

        return sorted_results, best_filter_params

    # ---------------------------------------------------------------------------------------------------------------#
    def kalmansmooth_search(self):
        # ======= I. Define the Kalman Smooth methods to test =======
        kf_smooth_coefficient = [0.5, 0.6, 0.7, 0.8, 0.9]  # False is the default value, it is tested in the initial params

        params_list = self.current_top_params.copy()

        # ------- Kalman Smooth coefficient : X*5 -------
        for current_top_params in self.current_top_params:
            params_kf_false = current_top_params.copy()
            params_kf_false["use_kf_weight"] = False
            params_list.append(params_kf_false)

            for kf in kf_smooth_coefficient:
                params = current_top_params.copy()
                params["use_kf_weight"] = True
                params["kf_smooth_coefficient"] = kf
                params_list.append(params)

        # ======= II. Run the simulations =======
        search_results = self.parallel_simulations(params_list)
        sorted_results = search_results.sort_values(by="sharpe_ratio", ascending=False)

        best_kalmansmooth_params = []
        for i in range(self.nb_paths_explored):
            hyper_params = sorted_results.iloc[i]["hyperparameters"]
            params = {**self.fixed_params, **hyper_params}
            best_kalmansmooth_params.append(params)

        # ======= III. Save the results =======
        self.evolution_history["kalmansmooth_search"] = search_results
        self.best_kalmansmooth_params = best_kalmansmooth_params
        self.current_top_params = best_kalmansmooth_params

        print("Kalman Smooth search completed.")

        return sorted_results, best_kalmansmooth_params

    # ---------------------------------------------------------------------------------------------------------------#
    def clustering_search(self):
        # ======= I. Define the Clustering methods to test =======
        linkages = ["DBHT", "median", "average"]  # "riskfolio" & "ward" is the default value, it is tested in the initial params
        n_clusters = [5, 8, 10, 13, 16, 20]

        params_list = self.current_top_params.copy()

        # ------- Linkage method : X*2 -------
        for current_top_params in self.current_top_params:
            for linkage in linkages:
                params = current_top_params.copy()
                params["clustering_method"] = "riskfolio"
                params["linkage"] = linkage
                params_list.append(params)

        # ------- DTW nb_clusters : X*6 -------
        for current_top_params in self.current_top_params:
            for n_cluster in n_clusters:
                params = current_top_params.copy()
                params["clustering_method"] = "dtw"
                params["n_clusters"] = n_cluster
                params_list.append(params)

        # ======= II. Run the simulations =======
        search_results = self.parallel_simulations(params_list)
        sorted_results = search_results.sort_values(by="sharpe_ratio", ascending=False)

        best_clustering_params = []
        for i in range(self.nb_paths_explored):
            hyper_params = sorted_results.iloc[i]["hyperparameters"]
            params = {**self.fixed_params, **hyper_params}
            best_clustering_params.append(params)

        # ======= III. Save the results =======
        self.evolution_history["clustering_search"] = search_results
        self.best_clustering_params = best_clustering_params
        self.current_top_params = best_clustering_params

        print("Clustering search completed.")

        return sorted_results, best_clustering_params

    # ---------------------------------------------------------------------------------------------------------------#
    def management_search(self):
        # ======= I. Define the Management methods to test =======
        training_windows = [180, 250, 550]  # 365 is the default value, it is tested in the initial params
        min_sharpe_to_new_portfolio = [0.5, 0.65, 1, 1.15]  # 0.8 is the default value, it is tested in the initial params
        min_sharpe_to_trade = [0.5, 0.65, 1, 1.15]  # 1 is the default value, it is tested in the initial params
        min_sharpe_to_rebalance = [0.5, 0.65, 1, 1.15, 1.3]  # 1.1 is the default value, it is tested in the initial params
        min_sharpe_spread_to_rebalance = [0, 0.1, 0.2, 0.35]  # 0 is the default value, it is tested in the initial params
        min_nb_pairs_to_new_portfolio = [2, 3, 4, 5]  # 0 is the default value, it is tested in the initial params
        min_nb_pairs_to_trade = [2, 3, 4, 5]  # 0 is the default value, it is tested in the initial params

        params_list = self.current_top_params.copy()

        # ------- Training window : X*5 -------
        for current_top_params in self.current_top_params:
            for window in training_windows:
                params = current_top_params.copy()
                params["training_window"] = window
                params_list.append(params)

        # ------- Min Sharpe to new portfolio : X*7 -------
        for current_top_params in self.current_top_params:
            for min_sharpe in min_sharpe_to_new_portfolio:
                params = current_top_params.copy()
                params["min_sharpe_to_new_portfolio"] = min_sharpe
                params_list.append(params)

        # ------- Min Sharpe to trade : X*7 -------
        for current_top_params in self.current_top_params:
            for min_sharpe in min_sharpe_to_trade:
                params = current_top_params.copy()
                params["min_sharpe_to_trade"] = min_sharpe
                params_list.append(params)

        # ------- Min Sharpe to rebalance : X*7 -------
        for current_top_params in self.current_top_params:
            for min_sharpe in min_sharpe_to_rebalance:
                params = current_top_params.copy()
                params["min_sharpe_to_rebalance"] = min_sharpe
                params_list.append(params)

        # ------- Min Sharpe spread to rebalance : X*4 -------
        for current_top_params in self.current_top_params:
            for min_sharpe in min_sharpe_spread_to_rebalance:
                params = current_top_params.copy()
                params["min_sharpe_spread_to_rebalance"] = min_sharpe
                params_list.append(params)

        # ------- Min nb pairs to new portfolio : X*3 -------
        for current_top_params in self.current_top_params:
            for min_nb in min_nb_pairs_to_new_portfolio:
                params = current_top_params.copy()
                params["min_nb_pairs_to_new_portfolio"] = min_nb
                params_list.append(params)

        # ------- Min nb pairs to trade : X*3 -------
        for current_top_params in self.current_top_params:
            for min_nb in min_nb_pairs_to_trade:
                params = current_top_params.copy()
                params["min_nb_pairs_to_trade"] = min_nb
                params_list.append(params)

        # ======= II. Run the simulations =======
        search_results = self.parallel_simulations(params_list)
        sorted_results = search_results.sort_values(by="sharpe_ratio", ascending=False)

        best_management_params = []
        for i in range(self.nb_paths_explored):
            hyper_params = sorted_results.iloc[i]["hyperparameters"]
            params = {**self.fixed_params, **hyper_params}
            best_management_params.append(params)

        # ======= III. Save the results =======
        self.evolution_history["management_search"] = search_results
        self.best_management_params = best_management_params
        self.current_top_params = best_management_params

        print("Management search completed.")

        return sorted_results, best_management_params
