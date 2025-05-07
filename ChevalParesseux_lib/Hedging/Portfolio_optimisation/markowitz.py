from ..Portfolio_optimisation import common as com

import numpy as np
import pandas as pd
from typing import Union
from typing import Self

from sklearn.covariance import LedoitWolf
from cvxpy import Variable, Minimize, Problem, quad_form, sum_entries


class markowitz_portfolio(com.Portfolio):
    #?_____________________________ Initialization methods _______________________________ #
    def __init__(self, n_jobs: int = 1) -> None:
        """
        Constructor for the MarkowitzPortfolio class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to use during feature computation.
        """
        super().__init__(n_jobs=n_jobs)
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
        rebalancing_window: int,
        risk_metrics: str,
    ) -> Self:
        
        self.params = {
            "rebalancing_window": rebalancing_window,
            "risk_metrics": risk_metrics,
        }

        return self
    
    #?________________________________ Auxiliary methods _________________________________ #
    def process_data(
        self, 
        data: Union[tuple, pd.DataFrame],
    ):
        # ======= I. Extract Series =======
        if isinstance(data, tuple):
            processed_data = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            processed_data = data.copy()
        else:
            raise ValueError("Input data must be a tuple or a pandas DataFrame.")

        return processed_data
    
    #?____________________________________________________________________________________ #
    def get_portfolio(
        self, 
        data: Union[tuple, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Construct the Markowitz mean-variance optimized portfolio.

        Parameters:
            - data (tuple or DataFrame): Historical price data (time x assets)

        Returns:
            - pd.Series: Portfolio weights indexed by asset names.
        """
        # ======= I. Prepare the data =======
        price_df = self.process_data(data)
        returns = price_df.pct_change().dropna()

        # ======= II. Estimate parameters =======
        mu = returns.mean()  # Expected return vector

        if self.params["risk_metrics"] == "sample_cov":
            Sigma = returns.cov()  # Sample covariance
        elif self.params["risk_metrics"] == "ledoit_wolf":
            Sigma = LedoitWolf().fit(returns).covariance_
            Sigma = pd.DataFrame(Sigma, index=returns.columns, columns=returns.columns)
        else:
            raise ValueError("Unsupported risk metric. Choose 'sample_cov' or 'ledoit_wolf'.")

        n = len(mu)
        w = Variable(n)

        # ======= III. Optimization problem =======
        objective = Minimize(quad_form(w, Sigma))  # Minimize risk
        constraints = [sum_entries(w) == 1, w >= 0]  # Fully invested, long-only

        prob = Problem(objective, constraints)
        prob.solve()

        weights = np.array(w.value).flatten()
        weights_series = pd.Series(weights, index=mu.index)
        return weights_series