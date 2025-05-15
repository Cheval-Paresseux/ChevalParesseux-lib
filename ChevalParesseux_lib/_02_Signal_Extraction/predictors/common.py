from ...utils import metrics as met

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, Self, Optional
from joblib import Parallel, delayed



#! ==================================================================================== #
#! =================================== Base Models ==================================== #
class Model(ABC):
    """
    This class defines the core structure and interface for models. It is meant to be subclassed
    by specific model implementations.
    
    Subclasses must implement the following abstract methods:
        - __init__: Initializes the model with number of jobs.
        - set_params: Defines the parameters.
        - process_data: Applies preprocessing to the data.
        - fit: Fits the model to the training data.
        - predict: Makes predictions on the test data.
    """
    #?_____________________________ Initialization methods _______________________________ #
    @abstractmethod
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        """
        Initializes the Model object.

        Parameters:
            - n_jobs (int): Number of parallel jobs to use for computation.
        """
        # ======= I. Initialize Class =======
        self.n_jobs = n_jobs

        # ======= II. Initialize Auxilaries =======
        self.params = {}
        self.metrics = {}
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def set_params(
        self,
        **kwargs
    ) -> Self:
        """
        Sets the parameter for the model.

        Parameters:
            - **kwargs: Additional parameters to be set.

        Returns:
            - Self: The instance of the class with the parameter set.
        """
        ...

    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(
        self,
        data: Union[tuple, pd.Series, pd.DataFrame],
        **kwargs
    ) -> Union[tuple, pd.DataFrame, pd.Series]:
        """
        Preprocesses the data.

        Parameters:
            - data (tuple | pd.Series | pd.DataFrame): The input data to be processed.
            - **kwargs: Additional parameters for the data processing.

        Returns:
            - tuple or pd.DataFrame or pd.Series: The processed data ready for extraction.
        """
        ...

    #?________________________________ Auxiliary methods _________________________________ #
    @abstractmethod
    def fit(
        self,
        X_train: Union[pd.DataFrame, pd.Series],
        y_train: pd.Series,
        **kwargs
    ) -> Self:
        """
        Fit the model to the training data.
        
        Parameters:
            - X_train (pd.DataFrame | pd.Series): The input features for training.
            - y_train (pd.Series): The target variable for training.
            - **kwargs: Additional parameters for fitting the model.
        
        Returns:
            - None
        """
        ...
    
    #?_________________________________ Callable methods _________________________________ #
    @abstractmethod
    def predict(
        self,
        X_test: Union[pd.DataFrame, pd.Series],
        **kwargs
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Makes predictions on the test data.
        
        Parameters:
            - X_test (pd.DataFrame | pd.Series): The input features for testing.
            - **kwargs: Additional parameters for making predictions.
        
        Returns:
            - pd.DataFrame or pd.Series: The predicted values.
        """
        ...

    #?__________________________________ Common methods __________________________________ #
    def get_learning_rate(
        self,
        learning_rate: float, 
        current_loss: float, 
        last_loss: float,
        increasing_speed: float = 0.5,
        decreasing_speed: float = 1.05
    ) -> float:
        """
        Computes the new learning rate based on the loss.
        
        Parameters:
            - learning_rate (float): The current learning rate.
            - current_loss (float): The current loss value.
            - last_loss (float): The previous loss value.
        
        Returns:
            - float: The updated learning rate.
        """
        # ----- 1. The loss is increasing -----
        if current_loss > last_loss:
            speed = increasing_speed
        
        # ----- 2. The loss is decreasing -----
        else:
            speed = decreasing_speed
        
        # ----- 3. Compute the new learning rate -----
        new_rate = learning_rate * speed
        
        return new_rate
    
    #?____________________________________________________________________________________ #
    def get_early_stopping(
        self,
        current_loss: float, 
        last_loss: float,
        threshold: float = 1e-5
    ) -> bool:
        """
        Checks if the model should stop training based on the loss difference.
        
        Parameters:
            - current_loss (float): The current loss value.
            - last_loss (float): The previous loss value.
            - threshold (float): The threshold for early stopping.
        
        Returns:
            - bool: True if early stopping is triggered, False otherwise.
        """
        # ======= I. Check the loss diference =======
        if last_loss == np.inf:
            return False
        
        loss_diff = np.abs(current_loss - last_loss)
        early_stop = False
        
        # ======= II. Check if the loss difference is small enough =======
        if loss_diff < threshold:
            early_stop = True
        
        return early_stop
    
    #?____________________________________________________________________________________ #
    def get_regression_metrics(
        self,
        predictions: np.array,
        features_matrix: np.array,
        y_true: np.array, 
        coefficients: np.array,
        feature_names: Optional[list] = None
    ):
        """
        Computes regression metrics for the model predictions.
        
        Parameters:
            - predictions (np.array): The predicted values.
            - features_matrix (np.array): The training feature matrix.
            - y_true (np.array): The actual target values.
            - coefficients (np.array): The model coefficients.
            - feature_names (list, optional): The names of the features.
        
        Returns:
            - dict: A dictionary containing various regression metrics.
        """
        # ======= I. Prediction Accuracy =======
        rmse = met.get_regression_rmse(predictions, y_true)
        mse = met.get_regression_mse(predictions, y_true)
        smape = met.get_regression_smape(predictions, y_true)
        max_error = met.get_regression_max_error(predictions, y_true)
        
        # ======= II. Significance Measures =======
        r2 = met.get_regression_r2(predictions, y_true)
        significance_df = met.get_regression_significance(predictions, features_matrix, y_true, coefficients, feature_names)
        
        # ======= III. Residuals Measures =======
        residuals = y_true - predictions
        durbin_watson = met.get_durbin_watson(residuals)
        JB_stat, JB_p_value = met.get_jarque_bera(residuals)
        lm_stat, lm_p_value = met.breusch_pagan_test(features_matrix, residuals)
        
        # ======= IV. Create the metrics dictionary =======
        metrics_dict = {
            "rmse": rmse,
            "mse": mse,
            "smape": smape,
            "max_error": max_error,
            "r2": r2,
            "significance": significance_df,
            "durbin_watson": durbin_watson,
            "JB_stat": (JB_stat, JB_p_value),
            "lm_stat": (lm_stat, lm_p_value)
        }
        
        return metrics_dict
    
    #?____________________________________________________________________________________ #
