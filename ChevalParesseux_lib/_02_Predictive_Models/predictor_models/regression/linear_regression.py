from ..regression import common as com

import pandas as pd
import numpy as np
from typing import Union, Self, Optional



#! ==================================================================================== #
#! ================================ Regression Models ================================= #
class OLS_regression(com.Regression_Model):
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
class MSE_regression(com.Regression_Model):
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
class Ridge_regression(com.Regression_Model):
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
class Lasso_regression(com.Regression_Model):
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
class ElasticNet_regression(com.Regression_Model):
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
    
