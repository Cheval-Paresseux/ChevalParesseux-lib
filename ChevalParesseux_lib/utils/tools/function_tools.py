import numpy as np
import pandas as pd

from itertools import product
from dateutil.relativedelta import relativedelta
import random
import inspect



#! ==================================================================================== #
#! =========================== Dict-Function Interface ================================ #
def get_dict_universe(
    params_grid: dict
) -> list:
    """
    Generate all combinations of parameters from a grid dictionary.

    Parameters:
        - params_grid (dict): A dictionary where keys are parameter names and values are lists of possible values.

    Returns:
        - params_as_list (List[dict]): A list of dictionaries, each representing a unique combination of parameters.
    """
    # ======= I. Extract keys =======
    keys = list(params_grid.keys())

    # ======= II. Extract every possible tuples =======
    values_product = product(*params_grid.values())

    # ======= III. Create a list of dictionaries with the keys and values =======
    params_as_list = [dict(zip(keys, values)) for values in values_product]
    
    return params_as_list

#*____________________________________________________________________________________ #
def get_random_dict_universe(
    params_grid: dict,
    n_samples: int = 10,
    random_state: int = 72,
) -> list:
    """
    Generate random combinations of parameters from a grid dictionary.

    Parameters:
        - params_grid (dict): A dictionary where keys are parameter names and values are lists of possible values.

    Returns:
        - params_as_list (List[dict]): A list of dictionaries, each representing a unique combination of parameters.
    """
    # ======= I. Extract keys =======
    keys = list(params_grid.keys())

    # ======= II. Extract every possible tuples =======
    values_product = product(*params_grid.values())

    # ======= III. Create a list of dictionaries with the keys and values =======
    params_as_list = [dict(zip(keys, values)) for values in values_product]
    
    # ======= IV. Randomly select n_samples from the list =======
    random.seed(random_state)
    random.shuffle(params_as_list)
    
    random_params_as_list = params_as_list[:n_samples]
    
    return random_params_as_list

#*____________________________________________________________________________________ #
def get_func_params(
    func: callable, 
    param_dict: dict
) -> dict:
    """
    Filter the parameters of a function to only include those that are valid for the function's signature.
    
    Parameters:
        - func (callable): The function whose parameters are to be filtered.
        - param_dict (dict): A dictionary of parameters to filter.
    
    Returns:
        - valid_params (dict): A dictionary containing only the parameters that are valid for the function.
    """
    # ======= I. Extract the right keys for the function =======
    sig = inspect.signature(func)
    valid_keys = sig.parameters.keys()

    # ======= II. Filter the parameters =======
    valid_params = {k: v for k, v in param_dict.items() if k in valid_keys}
    
    return valid_params

#*____________________________________________________________________________________ #
def get_walkforward_dates(
    start_date: str, 
    end_date: str, 
    train_period: str = '4Y', 
    test_period: str = '6M'
) -> tuple:
    """
    Generate walk-forward dates for training and testing periods.
    
    Parameters:
        - start_date (str): The start date of the dataset in 'YYYY-MM-DD' format.
        - end_date (str): The end date of the dataset in 'YYYY-MM-DD' format.
        - train_period (str): The training period in the format '6M', '1Y', etc. (default is '4Y').
        - test_period (str): The testing period in the format '6M', '1Y', etc. (default is '6M').
    
    Returns:
        - train_starts (list): List of start dates for each training period.
        - train_ends (list): List of end dates for each training period.
        - test_ends (list): List of end dates for each testing period.
    """
    # ======= O. Helper function to parse the period strings =======
    def parse_period(p):
        """
        Parse a period string like '6M' or '1Y' into a relativedelta object.
        
        Parameters:
            - p (str): The period string to parse.
        
        Returns:
            - relativedelta: A relativedelta object representing the period.
        """
        # ======= I. Extract Number and Unit =======
        n = int(p[:-1])
        unit = p[-1].upper()
        
        # ======= II. Convert to relativedelta =======
        if unit == 'M':
            return relativedelta(months=n)
        
        elif unit == 'Y':
            return relativedelta(years=n)
        
        else:
            raise ValueError("Unsupported period format. Use '6M', '1Y', etc.")

    # ======= I. Ensure inoput format =======
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # ======= II. Extract relativedelta object =======
    train_delta = parse_period(train_period)
    test_delta = parse_period(test_period)

    # ======= III. Initialize lists to store the dates =======
    train_starts, train_ends, test_ends = [], [], []
    
    # ======= IV. Generate walk-forward dates =======
    current_train_start = start_date
    while True:
        train_end = current_train_start + train_delta
        test_end = train_end + test_delta

        if test_end > end_date:
            test_end = end_date
            
            train_starts.append(current_train_start.strftime('%Y-%m-%d'))
            train_ends.append(train_end.strftime('%Y-%m-%d'))
            test_ends.append(test_end.strftime('%Y-%m-%d'))

            current_train_start += test_delta
            break

        train_starts.append(current_train_start.strftime('%Y-%m-%d'))
        train_ends.append(train_end.strftime('%Y-%m-%d'))
        test_ends.append(test_end.strftime('%Y-%m-%d'))

        current_train_start += test_delta  # Slide forward by test period

    return train_starts, train_ends, test_ends
