import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class ForwardLike(ABC):
    def __init__(self, subjacent: str = None, quantity: float = None, maturity: float|str = None, location: str = None):
        # ======= I. Contract Caracteristics =======
        self.subjacent = subjacent
        self.quantity = quantity
        self.maturity = maturity
        self.location = location
        
        # ======= II. Contract Pricing =======
        self.forward_price = None
        self.contract_price = None
    
    @abstractmethod
    def forwardPrice(self):
        pass
        
        

# ==================================================================================== #
# ============================== Forward Price  ====================================== #
def forwardPrice_stocklike(
    spot_price: float,
    risk_free_rate: float,
    maturity: float,
    dividend_rate: float = 0,
    storing_cost: float = 0,
):
    forward_price = (spot_price + storing_cost) * np.exp((risk_free_rate - dividend_rate) * maturity)

    return forward_price

# ____________________________________________________________________________________ #
def forwardPrice_currency(
    spot_price_AB: float,
    rf_A: float,
    rf_B: float,
    maturity: float,
):
    forward_price = spot_price_AB * np.exp((rf_A - rf_B) * maturity)

    return forward_price

# ==================================================================================== #
# ============================== Hedging with Futures  =============================== #
def futures_hedging(
    position_to_hedge_size: float,
    futures_contracts_size: float,
    correlation: float,
    subjacent_standard_deviation: float,
    futures_standard_deviation: float,
):
    # ======= I. Calculate the hedge ratio =======
    hedge_ratio = correlation * subjacent_standard_deviation / futures_standard_deviation

    # ======= II. Calculate the number of futures contracts needed =======
    number_of_contracts_needed = hedge_ratio * position_to_hedge_size / futures_contracts_size

    return hedge_ratio, number_of_contracts_needed