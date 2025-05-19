from .. import metrics as met

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



#! ==================================================================================== #
#! ================================ Regression Models ================================= #
class OLS_regression(Model):
    """
    OLS Regression Model using Normal Equation.
    
    This class implements Ordinary Least Squares (OLS) regression using the normal equation method.
    It is designed to fit a linear model to the training data and make predictions on new data.
    """
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        """
        Constructor for OLS_regression class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to use for computation.
        """
        super().__init__(n_jobs=n_jobs)
        
        # --- Model Parameters ---
        self.coefficients = None
        self.intercept = None
        
        # --- Model Metrics ---
        self.metrics = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self,
    ) -> Self:
        """
        Sets the parameter grid for the model.
        
        Parameters:
            The OLS regression model does not require any specific parameters to be set.
        
        Returns:
            - Self: The instance of the class with the parameter set.
        """
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
    def process_data(
        self, 
        features_matrix: Union[pd.DataFrame, np.array],
        target_vector: Union[pd.Series, np.array]
    ) -> tuple:
        """
        Transforms the input data into a suitable format for regression analysis.
        
        Parameters:
            - features_matrix (pd.DataFrame | np.array): The input features for training.
            - target_vector (pd.Series | np.array): The target variable for training.
        
        Returns:
            - tuple: A tuple containing the processed features and target variable.
        """
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(features_matrix).reshape(-1, 1) if len(np.array(features_matrix).shape) == 1 else np.array(features_matrix)
        y = np.array(target_vector)
        
        return X, y

    #?____________________________________________________________________________________ #
    def normal_equation(
        self, 
        features_matrix: np.array, 
        target_vector: np.array
    ) -> tuple:
        """
        Performs OLS regression using the normal equation method.
        
        Parameters:
            - features_matrix (np.array): The input features for training.
            - target_vector (np.array): The target variable for training.
        
        Returns:
            - tuple: A tuple containing the coefficients and intercept of the fitted model.
        """
        # ======= I. Add intercept to the features matrix =======
        X_with_intercept = np.c_[np.ones((features_matrix.shape[0], 1)), features_matrix]

        # ======= II. Compute coefficients using the normal equation =======
        XTX = X_with_intercept.T @ X_with_intercept
        XTy = X_with_intercept.T @ target_vector

        # ======= III. Solve for coefficients using the pseudo-inverse =======
        coefficients = np.linalg.pinv(XTX) @ XTy

        # ===== IV. Extract intercept and coefficients =======
        intercept = coefficients[0]
        coefficients = coefficients[1:]

        return coefficients, intercept
    
    #?_____________________________ User Functions _______________________________________ #
    def fit(
        self, 
        X_train: Union[pd.DataFrame, np.array],
        y_train: Union[pd.Series, np.array]
    ) -> Self:
        """
        Fit the OLS regression model to the training data.
        
        Parameters:
            - X_train (pd.DataFrame | np.array): The input features for training.
            - y_train (pd.Series | np.array): The target variable for training.
        
        Returns:
            - Self: The instance of the class with the fitted model.
        """
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.normal_equation(X, y)

        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        metrics = self.get_regression_metrics(
            predictions=train_predictions, 
            features_matrix=X, 
            y_true=y, 
            coefficients=self.coefficients,
            feature_names=None
        )
        self.metrics = metrics
        
        return self
        
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test: Union[pd.DataFrame, np.array]
    ) -> np.array:
        """
        Makes predictions using the fitted OLS regression model.
        
        Parameters:
            - X_test (pd.DataFrame | np.array): The input features for prediction.
        
        Returns:
            - np.array: The predicted values.
        """
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        
        return predictions
    
#*____________________________________________________________________________________ #
class MSE_regression(Model):
    """
    Gradient Descent Regression Model using Mean Squared Error (MSE) loss function.
    
    This class implements a linear regression model using gradient descent optimization.
    It is designed to fit a linear model to the training data and make predictions on new data.
    """
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        """
        Constructor for MSE_regression class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to use for computation.
        """
        super().__init__(n_jobs=n_jobs)
        
        # --- Model Parameters ---
        self.coefficients = None
        self.intercept = None
        
        # --- Model Metrics ---
        self.metrics = None
        self.loss_history = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        learning_rate: float = 0.01, 
        epochs: int = 1000
    ) -> Self:
        """
        Sets the parameter for the model.
        
        Parameters:
            - learning_rate (float): The learning rate for gradient descent.
            - epochs (int): The number of iterations for gradient descent.
        
        Returns:
            - Self: The instance of the class with the parameter set.
        """
        self.params = {
            'learning_rate': learning_rate,
            'epochs': epochs
        }
        
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
    def process_data(
        self, 
        features_matrix: Union[pd.DataFrame, np.array],
        target_vector: Union[pd.Series, np.array]
    ) -> tuple:
        """
        Transforms the input data into a suitable format for regression analysis.
        
        Parameters:
            - features_matrix (pd.DataFrame | np.array): The input features for training.
            - target_vector (pd.Series | np.array): The target variable for training.
        
        Returns:
            - tuple: A tuple containing the processed features and target variable.
        """
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(features_matrix).reshape(-1, 1) if len(np.array(features_matrix).shape) == 1 else np.array(features_matrix)
        y = np.array(target_vector)
        
        return X, y

    #?____________________________________________________________________________________ #
    def MSE_gradient(
        self, 
        nb_observations: int, 
        errors: np.array, 
        features_matrix: np.array
    ) -> tuple:
        """
        Computes the gradient of the Mean Squared Error (MSE) loss function.
        
        Parameters:
            - nb_observations (int): The number of observations in the dataset.
            - errors (np.array): The difference between predicted and actual values.
            - features_matrix (np.array): The input features for training.
        
        Returns:
            - tuple: A tuple containing the gradients for coefficients and intercept.
        """
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors)
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)
        
        return gradient_coefficients, gradient_intercept
    
    #?____________________________________________________________________________________ #
    def gradient_descent(
        self, 
        learning_rate: float, 
        epochs: int, 
        features_matrix: np.array, 
        target_vector: np.array
    ) -> tuple:
        """
        Computes the coefficients and intercept using gradient descent.
        
        Parameters:
            - learning_rate (float): The learning rate for gradient descent.
            - epochs (int): The number of iterations for gradient descent.
            - features_matrix (np.array): The input features for training.
            - target_vector (np.array): The target variable for training.
        
        Returns:
            - tuple: A tuple containing the coefficients and intercept of the fitted model.
        """
        # ======= I. Initialize coefficients and intercept to 0 =======
        learningRate = learning_rate
        nb_observations, nb_features = features_matrix.shape
        
        coefficients = np.zeros(nb_features)
        intercept = 0

        # ======= II. Perform gradient descent =======
        last_loss = np.inf
        loss_history = []
        for _ in range(epochs):
            # II.1 Make a prediction with the current coefficients and intercept
            predictions = intercept + np.dot(features_matrix, coefficients)

            # II.2 Compute the current errors
            errors = predictions - target_vector
            loss = np.sum(errors ** 2) / nb_observations
            loss_history.append(loss)
            
            # II.3 Update Learning Rate based on the loss
            learningRate = self.get_learning_rate(learningRate, loss, last_loss)
            early_stop = self.get_early_stopping(loss, last_loss)
            if early_stop:
                break
            last_loss = loss
            
            # II.4 Compute the gradient of the loss function
            gradient_coefficients, gradient_intercept = self.MSE_gradient(nb_observations, errors, features_matrix)

            # II.5 Update coefficients and intercept
            coefficients += learningRate * gradient_coefficients
            intercept += learningRate * gradient_intercept
        
        self.loss_history = loss_history

        return coefficients, intercept
    
    #?_____________________________ User Functions _______________________________________ #
    def fit(
        self, 
        X_train: Union[pd.DataFrame, np.array],
        y_train: Union[pd.Series, np.array]
    ) -> Self:
        """
        Fit the MSE regression model to the training data.
        
        Parameters:
            - X_train (pd.DataFrame | np.array): The input features for training.
            - y_train (pd.Series | np.array): The target variable for training.
        
        Returns:
            - Self: The instance of the class with the fitted model.
        """
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(self.params['learning_rate'], self.params['epochs'], X, y)
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        metrics = self.get_regression_metrics(
            predictions=train_predictions, 
            features_matrix=X, 
            y_true=y, 
            coefficients=self.coefficients,
            feature_names=None
        )
        self.metrics = metrics
        
        return self
        
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test: Union[pd.DataFrame, np.array]
    ) -> np.array:
        """
        Makes predictions using the fitted regression model.
        
        Parameters:
            - X_test (pd.DataFrame | np.array): The input features for prediction.
        
        Returns:
            - np.array: The predicted values.
        """
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        
        return predictions
    
#*____________________________________________________________________________________ #
class Ridge_regression(Model):
    """
    Gradient Descent Regression Model using Ridge Regularization.
    
    This class implements a linear regression model using gradient descent optimization.
    It is designed to fit a linear model to the training data and make predictions on new data.
    """
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        """
        Constructor for Ridge_regression class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to use for computation.
        """
        super().__init__(n_jobs=n_jobs)
        
        # --- Model Parameters ---
        self.coefficients = None
        self.intercept = None
        
        # --- Model Metrics ---
        self.metrics = None
        self.loss_history = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        lambda_: float = 0.1, 
        learning_rate: float = 0.01, 
        epochs: int = 1000
    ) -> Self:
        """
        Sets the parameter for the model.
        
        Parameters:
            - lambda_ (float): The regularization parameter for Ridge regression.
            - learning_rate (float): The learning rate for gradient descent.
            - epochs (int): The number of iterations for gradient descent.
        
        Returns:
            - Self: The instance of the class with the parameter set.
        """
        self.params = {
            'lambda': lambda_,
            'learning_rate': learning_rate,
            'epochs': epochs
        }
        
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
    def process_data(
        self, 
        features_matrix: Union[pd.DataFrame, np.array],
        target_vector: Union[pd.Series, np.array]
    ) -> tuple:
        """
        Transforms the input data into a suitable format for regression analysis.
        
        Parameters:
            - features_matrix (pd.DataFrame | np.array): The input features for training.
            - target_vector (pd.Series | np.array): The target variable for training.
        
        Returns:
            - tuple: A tuple containing the processed features and target variable.
        """
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(features_matrix).reshape(-1, 1) if len(np.array(features_matrix).shape) == 1 else np.array(features_matrix)
        y = np.array(target_vector)
        
        return X, y

    #?____________________________________________________________________________________ #
    def ridge_gradient(
        self, 
        nb_observations: int, 
        errors: np.array, 
        features_matrix: np.array, 
        lambda_: float, 
        coefficients: np.array
    ) -> tuple:
        """
        Computes the gradient of the Ridge loss function.
        
        Parameters:
            - nb_observations (int): The number of observations in the dataset.
            - errors (np.array): The difference between predicted and actual values.
            - features_matrix (np.array): The input features for training.
            - lambda_ (float): The regularization parameter for Ridge regression.
            - coefficients (np.array): The current coefficients of the model.
        
        Returns:
            - tuple: A tuple containing the gradients for coefficients and intercept.
        """
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors) + 2 * lambda_ * coefficients
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)

        return gradient_coefficients, gradient_intercept
    
    #?____________________________________________________________________________________ #
    def gradient_descent(
        self, 
        learning_rate: float, 
        epochs: int, 
        features_matrix: np.array, 
        target_vector: np.array,
        lambda_: float
    ) -> tuple:
        """
        Computes the coefficients and intercept using gradient descent.
        
        Parameters:
            - learning_rate (float): The learning rate for gradient descent.
            - epochs (int): The number of iterations for gradient descent.
            - features_matrix (np.array): The input features for training.
            - target_vector (np.array): The target variable for training.
        
        Returns:
            - tuple: A tuple containing the coefficients and intercept of the fitted model.
        """
        # ======= I. Initialize coefficients and intercept to 0 =======
        learningRate = learning_rate
        nb_observations, nb_features = features_matrix.shape
        
        coefficients = np.zeros(nb_features)
        intercept = 0

        # ======= II. Perform gradient descent =======
        last_loss = np.inf
        loss_history = []
        for _ in range(epochs):
            # II.1 Make a prediction with the current coefficients and intercept
            predictions = intercept + np.dot(features_matrix, coefficients)

            # II.2 Compute the current errors
            errors = predictions - target_vector
            loss = np.sum(errors ** 2) / nb_observations
            loss_history.append(loss)
            
            # II.3 Update Learning Rate based on the loss
            learningRate = self.get_learning_rate(learningRate, loss, last_loss)
            early_stop = self.get_early_stopping(loss, last_loss)
            if early_stop:
                break
            last_loss = loss
            
            # II.4 Compute the gradient of the loss function
            gradient_coefficients, gradient_intercept = self.ridge_gradient(nb_observations, errors, features_matrix, lambda_, coefficients)

            # II.5 Update coefficients and intercept
            coefficients += learningRate * gradient_coefficients
            intercept += learningRate * gradient_intercept
        
        self.loss_history = loss_history

        return coefficients, intercept
    
    #?_____________________________ User Functions _______________________________________ #
    def fit(
        self, 
        X_train: Union[pd.DataFrame, np.array],
        y_train: Union[pd.Series, np.array]
    ) -> Self:
        """
        Fit the Ridge regression model to the training data.
        
        Parameters:
            - X_train (pd.DataFrame | np.array): The input features for training.
            - y_train (pd.Series | np.array): The target variable for training.
        
        Returns:
            - Self: The instance of the class with the fitted model.
        """
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(
            learning_rate=self.params['learning_rate'], 
            epochs=self.params['epochs'], 
            features_matrix=X, 
            target_vector=y,
            lambda_=self.params['lambda'],
        )
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        metrics = self.get_regression_metrics(
            predictions=train_predictions, 
            features_matrix=X, 
            y_true=y, 
            coefficients=self.coefficients,
            feature_names=None
        )
        self.metrics = metrics
        
        return self
        
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test: Union[pd.DataFrame, np.array]
    ) -> np.array:
        """
        Makes predictions using the fitted regression model.
        
        Parameters:
            - X_test (pd.DataFrame | np.array): The input features for prediction.
        
        Returns:
            - np.array: The predicted values.
        """
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        
        return predictions
    
#*____________________________________________________________________________________ #
class Lasso_regression(Model):
    """
    Gradient Descent Regression Model using Lasso Regularization.
    
    This class implements a linear regression model using gradient descent optimization.
    It is designed to fit a linear model to the training data and make predictions on new data.
    """
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        """
        Constructor for Lasso_regression class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to use for computation.
        """
        super().__init__(n_jobs=n_jobs)
        
        # --- Model Parameters ---
        self.coefficients = None
        self.intercept = None
        
        # --- Model Metrics ---
        self.metrics = None
        self.loss_history = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        lambda_: float = 0.1, 
        learning_rate: float = 0.01, 
        epochs: int = 1000
    ) -> Self:
        """
        Sets the parameter for the model.
        
        Parameters:
            - lambda_ (float): The regularization parameter for Lasso regression.
            - learning_rate (float): The learning rate for gradient descent.
            - epochs (int): The number of iterations for gradient descent.
        
        Returns:
            - Self: The instance of the class with the parameter set.
        """
        self.params = {
            'lambda': lambda_,
            'learning_rate': learning_rate,
            'epochs': epochs
        }
        
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
    def process_data(
        self, 
        features_matrix: Union[pd.DataFrame, np.array],
        target_vector: Union[pd.Series, np.array]
    ) -> tuple:
        """
        Transforms the input data into a suitable format for regression analysis.
        
        Parameters:
            - features_matrix (pd.DataFrame | np.array): The input features for training.
            - target_vector (pd.Series | np.array): The target variable for training.
        
        Returns:
            - tuple: A tuple containing the processed features and target variable.
        """
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(features_matrix).reshape(-1, 1) if len(np.array(features_matrix).shape) == 1 else np.array(features_matrix)
        y = np.array(target_vector)
        
        return X, y

    #?____________________________________________________________________________________ #
    def lasso_gradient(
        self, 
        nb_observations: int, 
        errors: np.array, 
        features_matrix: np.array, 
        lambda_: float, 
        coefficients: np.array
    ) -> tuple:
        """
        Computes the gradient of the Lasso loss function.
        
        Parameters:
            - nb_observations (int): The number of observations in the dataset.
            - errors (np.array): The difference between predicted and actual values.
            - features_matrix (np.array): The input features for training.
            - lambda_ (float): The regularization parameter for Lasso regression.
            - coefficients (np.array): The current coefficients of the model.
        
        Returns:
            - tuple: A tuple containing the gradients for coefficients and intercept.
        """
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors) + lambda_ * np.sign(coefficients)
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)

        return gradient_coefficients, gradient_intercept
    
    #?____________________________________________________________________________________ #
    def gradient_descent(
        self, 
        learning_rate: float, 
        epochs: int, 
        features_matrix: np.array, 
        target_vector: np.array,
        lambda_: float
    ) -> tuple:
        """
        Computes the coefficients and intercept using gradient descent.
        
        Parameters:
            - learning_rate (float): The learning rate for gradient descent.
            - epochs (int): The number of iterations for gradient descent.
            - features_matrix (np.array): The input features for training.
            - target_vector (np.array): The target variable for training.
        
        Returns:
            - tuple: A tuple containing the coefficients and intercept of the fitted model.
        """
        # ======= I. Initialize coefficients and intercept to 0 =======
        learningRate = learning_rate
        nb_observations, nb_features = features_matrix.shape
        
        coefficients = np.zeros(nb_features)
        intercept = 0

        # ======= II. Perform gradient descent =======
        last_loss = np.inf
        loss_history = []
        for _ in range(epochs):
            # II.1 Make a prediction with the current coefficients and intercept
            predictions = intercept + np.dot(features_matrix, coefficients)

            # II.2 Compute the current errors
            errors = predictions - target_vector
            loss = np.sum(errors ** 2) / nb_observations
            loss_history.append(loss)
            
            # II.3 Update Learning Rate based on the loss
            learningRate = self.get_learning_rate(learningRate, loss, last_loss)
            early_stop = self.get_early_stopping(loss, last_loss)
            if early_stop:
                break
            last_loss = loss
            
            # II.4 Compute the gradient of the loss function
            gradient_coefficients, gradient_intercept = self.lasso_gradient(nb_observations, errors, features_matrix, lambda_, coefficients)

            # II.5 Update coefficients and intercept
            coefficients += learningRate * gradient_coefficients
            intercept += learningRate * gradient_intercept
        
        self.loss_history = loss_history

        return coefficients, intercept
    
    #?_____________________________ User Functions _______________________________________ #
    def fit(
        self, 
        X_train: Union[pd.DataFrame, np.array],
        y_train: Union[pd.Series, np.array]
    ) -> Self:
        """
        Fit the Lasso regression model to the training data.
        
        Parameters:
            - X_train (pd.DataFrame | np.array): The input features for training.
            - y_train (pd.Series | np.array): The target variable for training.
        
        Returns:
            - Self: The instance of the class with the fitted model.
        """
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(
            learning_rate=self.params['learning_rate'], 
            epochs=self.params['epochs'], 
            features_matrix=X, 
            target_vector=y,
            lambda_=self.params['lambda'],
        )

        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        metrics = self.get_regression_metrics(
            predictions=train_predictions, 
            features_matrix=X, 
            y_true=y, 
            coefficients=self.coefficients,
            feature_names=None
        )
        self.metrics = metrics
        
        return self
        
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test: Union[pd.DataFrame, np.array]
    ) -> np.array:
        """
        Makes predictions using the fitted regression model.
        
        Parameters:
            - X_test (pd.DataFrame | np.array): The input features for prediction.
        
        Returns:
            - np.array: The predicted values.
        """
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        
        return predictions
    
#*____________________________________________________________________________________ #
class ElasticNet_regression(Model):
    """
    Gradient Descent Regression Model using ElasticNet Regularization.
    It combines Lasso and Ridge regression.
    """
    def __init__(
        self, 
        n_jobs: int = 1
    ) -> None:
        """
        Constructor for ElasticNet_regression class.
        
        Parameters:
            - n_jobs (int): Number of parallel jobs to use for computation.
        """
        super().__init__(n_jobs=n_jobs)
        
        # --- Model Parameters ---
        self.coefficients = None
        self.intercept = None
        
        # --- Model Metrics ---
        self.metrics = None
        self.loss_history = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        lambda1: float = 0.1, 
        lambda2: float = 0.1, 
        learning_rate: float = 0.01, 
        epochs: int = 1000
    ) -> Self:
        """
        Sets the parameter for the model.
        
        Parameters:
            - lambda1 (float): The regularization parameter for Lasso regression.
            - lambda2 (float): The regularization parameter for Ridge regression.
            - learning_rate (float): The learning rate for gradient descent.
            - epochs (int): The number of iterations for gradient descent.
        
        Returns:
            - Self: The instance of the class with the parameter set.
        """
        self.params = {
            'lambda1': lambda1,
            'lambda2': lambda2,
            'learning_rate': learning_rate,
            'epochs': epochs
        }
        
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
    def process_data(
        self, 
        features_matrix: Union[pd.DataFrame, np.array],
        target_vector: Union[pd.Series, np.array]
    ) -> tuple:
        """
        Transforms the input data into a suitable format for regression analysis.
        
        Parameters:
            - features_matrix (pd.DataFrame | np.array): The input features for training.
            - target_vector (pd.Series | np.array): The target variable for training.
        
        Returns:
            - tuple: A tuple containing the processed features and target variable.
        """
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(features_matrix).reshape(-1, 1) if len(np.array(features_matrix).shape) == 1 else np.array(features_matrix)
        y = np.array(target_vector)
        
        return X, y

    #?____________________________________________________________________________________ #
    def elastic_net_gradient(
        self, 
        nb_observations: int, 
        errors: np.array, 
        features_matrix: np.array, 
        lambda1: float, 
        lambda2: float, 
        coefficients: np.array
    ) -> tuple:
        """
        Computes the gradient of the ElasticNet loss function.

        Parameters:
            - nb_observations (int): The number of observations in the dataset.
            - errors (np.array): The difference between predicted and actual values.
            - features_matrix (np.array): The input features for training.
            - lambda1 (float): The regularization parameter for Lasso regression.
            - lambda2 (float): The regularization parameter for Ridge regression.
            - coefficients (np.array): The current coefficients of the model.
        
        Returns:
            - tuple: A tuple containing the gradients for coefficients and intercept.
        """
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors) + 2 * lambda1 * coefficients + lambda2 * np.sign(coefficients)
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)

        return gradient_coefficients, gradient_intercept
    
    #?____________________________________________________________________________________ #
    def gradient_descent(
        self, 
        learning_rate: float, 
        epochs: int, 
        features_matrix: np.array, 
        target_vector: np.array, 
        lambda1: float,
        lambda2: float
    ) -> tuple:
        """
        Computes the coefficients and intercept using gradient descent.

        Parameters:
            - learning_rate (float): The learning rate for gradient descent.
            - epochs (int): The number of iterations for gradient descent.
            - features_matrix (np.array): The input features for training.
            - target_vector (np.array): The target variable for training.
            - lambda1 (float): The regularization parameter for Lasso regression.
            - lambda2 (float): The regularization parameter for Ridge regression.
        
        Returns:
            - tuple: A tuple containing the coefficients and intercept of the fitted model.
        """
        # ======= I. Initialize coefficients and intercept to 0 =======
        learningRate = learning_rate
        nb_observations, nb_features = features_matrix.shape
        
        coefficients = np.zeros(nb_features)
        intercept = 0

        # ======= II. Perform gradient descent =======
        last_loss = np.inf
        loss_history = []
        for _ in range(epochs):
            # II.1 Make a prediction with the current coefficients and intercept
            predictions = intercept + np.dot(features_matrix, coefficients)

            # II.2 Compute the current errors
            errors = predictions - target_vector
            loss = np.sum(errors ** 2) / nb_observations
            loss_history.append(loss)
            
            # II.3 Update Learning Rate based on the loss
            learningRate = self.get_learning_rate(learningRate, loss, last_loss)
            early_stop = self.get_early_stopping(loss, last_loss)
            if early_stop:
                break
            last_loss = loss
            
            # II.4 Compute the gradient of the loss function
            gradient_coefficients, gradient_intercept = self.elastic_net_gradient(nb_observations, errors, features_matrix, lambda1, lambda2, coefficients)

            # II.5 Update coefficients and intercept
            coefficients += learningRate * gradient_coefficients
            intercept += learningRate * gradient_intercept
        
        self.loss_history = loss_history

        return coefficients, intercept
    
    #?_____________________________ User Functions _______________________________________ #
    def fit(
        self, 
        X_train: Union[pd.DataFrame, np.array],
        y_train: Union[pd.Series, np.array]
    ) -> Self:
        """
        Fit the MSE regression model to the training data.
        
        Parameters:
            - X_train (pd.DataFrame | np.array): The input features for training.
            - y_train (pd.Series | np.array): The target variable for training.
        
        Returns:
            - Self: The instance of the class with the fitted model.
        """
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(
            learning_rate=self.params['learning_rate'], 
            epochs=self.params['epochs'], 
            features_matrix=X, 
            target_vector=y,
            lambda1=self.params['lambda1'],
            lambda2=self.params['lambda2']
        )
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        metrics = self.get_regression_metrics(
            predictions=train_predictions, 
            features_matrix=X, 
            y_true=y, 
            coefficients=self.coefficients,
            feature_names=None
        )
        self.metrics = metrics
        
        return self
        
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test: Union[pd.DataFrame, np.array]
    ) -> np.array:
        """
        Makes predictions using the fitted regression model.
        
        Parameters:
            - X_test (pd.DataFrame | np.array): The input features for prediction.
        
        Returns:
            - np.array: The predicted values.
        """
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        
        return predictions
    
