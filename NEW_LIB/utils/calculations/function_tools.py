import numpy as np
import pandas as pd

from itertools import product
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
