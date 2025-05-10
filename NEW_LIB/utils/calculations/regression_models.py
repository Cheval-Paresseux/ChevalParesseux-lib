import numpy as np
import pandas as pd
from typing import Union, Self
from scipy.stats import t



#! ==================================================================================== #
#! ================================= Helper Functions ================================= #
def adapt_learning_rate(
    learning_rate: float, 
    loss: float, 
    last_loss: float
) -> float:
    """
    Computes the new learning rate based on the loss difference.
    
    Parameters:
        - learning_rate (float): The current learning rate.
        - loss (float): The current loss value.
        - last_loss (float): The previous loss value.
    
    Returns:
        - new_rate (float): The updated learning rate.
    """
    new_rate = learning_rate
    if loss > last_loss:
        new_rate /= 2
    else:
        new_rate *= 1.05
    
    return new_rate

#*____________________________________________________________________________________ #
def early_stopping(
    loss: float, 
    last_loss: float
) -> bool:
    """
    Computes if the model should stop training based on the loss difference.
    
    Parameters:
        - loss (float): The current loss value.
        - last_loss (float): The previous loss value.
    
    Returns:
        - early_stop (bool): True if the model should stop training, False otherwise.
    """
    # ======= I. Check the loss diference =======
    if last_loss == np.inf:
        return False
    
    loss_diff = np.abs(loss - last_loss)
    early_stop = False
    
    # ======= II. Check if the loss difference is small enough =======
    if loss_diff < 1e-5:
        early_stop = True
    
    return early_stop

#*____________________________________________________________________________________ #
def get_regression_stats(
    predictions: np.array, 
    X_train: np.array, 
    y_train: np.array, 
    coefficients: np.array
) -> tuple:
    """
    Computes regression statistics including R-squared, variance, and p-values.
    
    Parameters:
        - predictions (np.array): The predicted values.
        - X_train (np.array): The input features matrix.
        - y_train (np.array): The target variable vector.
        - coefficients (np.array): The model coefficients.
    
    Returns:
        - statistics (dict): A dictionary containing the computed statistics.
        - residuals (np.array): The residuals of the model.
    """
    # ======= I. Compute Residuals =======
    residuals = y_train - predictions
    
    # ======= II. Compute Residual Statistics =======
    nb_observations, nb_features = X_train.shape

    if nb_observations <= nb_features:
        raise ValueError("Number of observations must be greater than the number of features to compute statistics.")

    variance = np.sum(residuals**2) / (nb_observations - nb_features)
    mean = np.mean(residuals)
    median = np.median(residuals)

    # ======= III. Compute R-Squared =======
    SST = np.sum((y_train - np.mean(y_train))**2)
    SSR = np.sum((predictions - np.mean(y_train))**2)
    R_squared = SSR / SST if SST != 0 else 0
    
    # ======= IV. Compute t-Statistics and p-Values =======
    XTX = X_train.T @ X_train

    # Use pseudo-inverse to avoid singularity issues
    var_covar_matrix = variance * np.linalg.pinv(XTX)
    se_coefficients = np.sqrt(np.diag(var_covar_matrix))
    t_stats = coefficients / se_coefficients if np.all(se_coefficients != 0) else np.zeros_like(coefficients)

    # Degrees of freedom check
    degrees_freedom = nb_observations - nb_features
    p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_freedom)) for t_stat in t_stats]

    # ======= V. Store the Statistics =======
    statistics = {
        "Variance": variance,
        "Mean": mean,
        "Median": median,
        "R_squared": R_squared,
        "T_stats": t_stats.tolist(),
        "P_values": p_values
    }

    return statistics, residuals



#! ==================================================================================== #
#! ================================ Regression Models ================================= #
class OLSRegression():
    def __init__(
        self, 
    ) -> None:
        # --- Data Fitted ---
        self.X_train = None
        self.y_train = None
        
        # --- Model Parameters ---
        self.params = None
        self.coefficients = None
        self.intercept = None
        
        # --- Model Statistics ---
        self.statistics = None
        self.residuals = None
    
    #?_____________________________ Build Functions ______________________________________ #
    def process_data(
        self, 
        X_train: Union[np.array, pd.DataFrame, pd.Series], 
        y_train: Union[np.array, pd.Series]
    ):
        """
        Transforms the input data into numpy arrays and stores them as class attributes.
        
        Parameters:
            - X_train (Union[np.array, pd.DataFrame, pd.Series]): The input features.
            - y_train (Union[np.array, pd.Series]): The target variable.
        
        Returns:
            - X (np.array): The transformed input features as a numpy array.
            - y (np.array): The transformed target variable as a numpy array.
        """
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(X_train).reshape(-1, 1) if len(np.array(X_train).shape) == 1 else np.array(X_train)
        y = np.array(y_train)
        
        # ======= II. Store training data =======
        self.X_train = X
        self.y_train = y
        
        return X, y

    #?____________________________________________________________________________________ #
    def normal_equation(
        self, 
        features_matrix: np.array, 
        target_vector: np.array
    ) -> tuple:
        """
        Computes the coefficients and intercept using the normal equation method.
        
        Parameters:
            - features_matrix (np.array): The input features matrix.
            - target_vector (np.array): The target variable vector.
        
        Returns:
            - coefficients (np.array): The computed coefficients.
            - intercept (float): The computed intercept.
        """
        # ======= I. Add intercept to the features matrix =======
        X_with_intercept = np.c_[np.ones((features_matrix.shape[0], 1)), features_matrix]

        # ======= II. Compute the coefficients using the normal equation =======
        XTX = X_with_intercept.T @ X_with_intercept
        XTy = X_with_intercept.T @ target_vector

        # ======= III. Solve for coefficients =======
        coefficients = np.linalg.pinv(XTX) @ XTy

        # ======= IV. Extract intercept and coefficients =======
        intercept = coefficients[0]
        coefficients = coefficients[1:]

        return coefficients, intercept
    
    #?_____________________________ User Functions _______________________________________ #
    def fit(
        self, 
        X_train: Union[np.array, pd.DataFrame, pd.Series],
        y_train: Union[np.array, pd.Series]
    ):
        """
        Computes the coefficients and intercept using the normal equation method.
        
        Parameters:
            - X_train (Union[np.array, pd.DataFrame, pd.Series]): The input features.
            - y_train (Union[np.array, pd.Series]): The target variable.
        
        Returns:
            - statistics (dict): A dictionary containing the computed statistics.
        """
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.normal_equation(X, y)
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        self.statistics, self.residuals = get_regression_stats(train_predictions, X, y, self.coefficients)
        
        return self.statistics
        
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test: Union[np.array, pd.DataFrame, pd.Series]
    ) -> np.array:
        """
        Computes predictions using the fitted model.
        
        Parameters:
            - X_test (Union[np.array, pd.DataFrame, pd.Series]): The input features for prediction.
        
        Returns:
            - predictions (np.array): The computed predictions.
        """
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        
        return predictions
    
#*____________________________________________________________________________________ #
class MSERegression():
    def __init__(
        self,
    ) -> None:
        # --- Data Fitted ---
        self.X_train = None
        self.y_train = None

        # --- Model Parameters ---
        self.params = None
        self.coefficients = None
        self.intercept = None
        
        # --- Model Statistics ---
        self.statistics = None
        self.residuals = None
        self.loss_history = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        learning_rate: float = 0.01, 
        epochs: int = 1000
    ) -> Self:
        """
        Sets the parameters for the model.
        
        Parameters:
            - learning_rate (float): The learning rate for gradient descent.
            - epochs (int): The number of iterations for gradient descent.
        
        Returns:
            - self (Self): The instance of the class with updated parameters.
        """
        self.params = {
            'learning_rate': learning_rate,
            'epochs': epochs
        }
        
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
    def process_data(
        self, 
        X_train: Union[np.array, pd.DataFrame, pd.Series],
        y_train: Union[np.array, pd.Series]
    ) -> tuple:
        """
        Transforms the input data into numpy arrays and stores them as class attributes.
        
        Parameters:
            - X_train: The input features.
            - y_train: The target variable.
            
        Returns:
            - X (np.array): The transformed input features as a numpy array.
            - y (np.array): The transformed target variable as a numpy array.
        """
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(X_train).reshape(-1, 1) if len(np.array(X_train).shape) == 1 else np.array(X_train)
        y = np.array(y_train)
        
        # ======= II. Store training data =======
        self.X_train = X
        self.y_train = y
        
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
            - errors (np.array): The errors between predictions and target values.
            - features_matrix (np.array): The input features matrix.
        
        Returns:
            - gradient_coefficients (np.array): The computed gradient for the coefficients.
            - gradient_intercept (float): The computed gradient for the intercept.
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
        Performs gradient descent to optimize the coefficients and intercept.
        
        Parameters:
            - learning_rate (float): The learning rate for gradient descent.
            - epochs (int): The number of iterations for gradient descent.
            - features_matrix (np.array): The input features matrix.
            - target_vector (np.array): The target variable vector.
        
        Returns:
            - coefficients (np.array): The optimized coefficients.
            - intercept (float): The optimized intercept.
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
            learningRate = adapt_learning_rate(learningRate, loss, last_loss)
            early_stop = early_stopping(loss, last_loss)
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
        X_train: Union[np.array, pd.DataFrame, pd.Series],
        y_train: Union[np.array, pd.Series]
    ) -> dict:
        """
        Computes the coefficients and intercept using gradient descent.
        
        Parameters:
            - X_train (Union[np.array, pd.DataFrame, pd.Series]): The input features.
            - y_train (Union[np.array, pd.Series]): The target variable.
        
        Returns:
            - statistics (dict): A dictionary containing the computed statistics.
        """
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(self.params['learning_rate'], self.params['epochs'], X, y)
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        self.statistics, self.residuals = get_regression_stats(train_predictions, X, y, self.coefficients)
        
        return self.statistics
        
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test: Union[np.array, pd.DataFrame, pd.Series]
    ) -> np.array:
        """
        Computes predictions using the fitted model.
        
        Parameters:
            - X_test (Union[np.array, pd.DataFrame, pd.Series]): The input features for prediction.
        
        Returns:
            - predictions (np.array): The computed predictions.
        """
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        
        return predictions
    
#*____________________________________________________________________________________ #
class RidgeRegression():
    def __init__(
        self, 
    ) -> None:        
        # --- Data Fitted ---
        self.X_train = None
        self.y_train = None
        
        # --- Model Parameters ---
        self.params = None
        self.coefficients = None
        self.intercept = None
        
        # --- Model Statistics ---
        self.statistics = None
        self.residuals = None
        self.loss_history = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        lambda_: float = 0.1, 
        learning_rate: float = 0.01, 
        epochs: int = 1000
    ) -> Self:
        self.params = {
            'lambda': lambda_,
            'learning_rate': learning_rate,
            'epochs': epochs
        }
        
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
    def process_data(
        self, 
        X_train, 
        y_train
    ) -> tuple:
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(X_train).reshape(-1, 1) if len(np.array(X_train).shape) == 1 else np.array(X_train)
        y = np.array(y_train)
        
        # ======= II. Store training data =======
        self.X_train = X
        self.y_train = y
        
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
        
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors) + 2 * lambda_ * coefficients
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
        # ======= I. Initialize coefficients and intercept to 0 =======
        learningRate = learning_rate
        nb_observations, nb_features = features_matrix.shape
        lambda_ = self.params['lambda']
        
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
            learningRate = adapt_learning_rate(learningRate, loss, last_loss)
            early_stop = early_stopping(loss, last_loss)
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
        X_train, 
        y_train
    ) -> dict:
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(self.params['learning_rate'], self.params['epochs'], X, y)
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        self.statistics, self.residuals = get_regression_stats(train_predictions, X, y, self.coefficients)
        
        return self.statistics
        
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test
    ) -> np.array:
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        
        return predictions
    
#*____________________________________________________________________________________ #
class LassoRegression():
    def __init__(
        self, 
    ) -> None:
        # --- Data Fitted ---
        self.X_train = None
        self.y_train = None

        # --- Model Parameters ---
        self.params = None
        self.coefficients = None
        self.intercept = None
        
        # --- Model Statistics ---
        self.statistics = None
        self.residuals = None
        self.loss_history = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        lambda_: float = 0.1, 
        learning_rate: float = 0.01, 
        epochs: int = 1000
    ) -> Self:
        self.params = {
            'lambda': lambda_,
            'learning_rate': learning_rate,
            'epochs': epochs
        }
        
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
    def process_data(
        self, 
        X_train, 
        y_train
    ) -> tuple:
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(X_train).reshape(-1, 1) if len(np.array(X_train).shape) == 1 else np.array(X_train)
        y = np.array(y_train)
        
        # ======= II. Store training data =======
        self.X_train = X
        self.y_train = y
        
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
        
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors) + lambda_ * np.sign(coefficients)
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
        # ======= I. Initialize coefficients and intercept to 0 =======
        learningRate = learning_rate
        nb_observations, nb_features = features_matrix.shape
        lambda_ = self.params['lambda']
        
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
            learningRate = adapt_learning_rate(learningRate, loss, last_loss)
            early_stop = early_stopping(loss, last_loss)
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
        X_train, 
        y_train
    ) -> dict:
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(self.params['learning_rate'], self.params['epochs'], X, y)
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        self.statistics, self.residuals = com.get_regression_stats(train_predictions, X, y, self.coefficients)
        
        return self.statistics
        
    #?____________________________________________________________________________________ #
    def predict(
        self, 
        X_test
    ) -> np.array:
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        
        return predictions
    
#*____________________________________________________________________________________ #
class ElasticNetRegression():
    def __init__(self) -> None:
        # --- Data Fitted ---
        self.X_train = None
        self.y_train = None

        # --- Model Parameters ---
        self.params = None
        self.coefficients = None
        self.intercept = None
        
        # --- Model Statistics ---
        self.statistics = None
        self.residuals = None
        self.loss_history = None
    
    #?____________________________________________________________________________________ #
    def set_params(
        self, 
        lambda1: float = 0.1, 
        lambda2: float = 0.1, 
        learning_rate: float = 0.01, 
        epochs: int = 1000
    ) -> Self:
        self.params = {
            'lambda1': lambda1,
            'lambda2': lambda2,
            'learning_rate': learning_rate,
            'epochs': epochs
        }
        
        return self
        
    #?_____________________________ Build Functions ______________________________________ #
    def process_data(self, X_train, y_train):
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(X_train).reshape(-1, 1) if len(np.array(X_train).shape) == 1 else np.array(X_train)
        y = np.array(y_train)
        
        # ======= II. Store training data =======
        self.X_train = X
        self.y_train = y
        
        return X, y

    #?____________________________________________________________________________________ #
    def elastic_net_gradient(self, nb_observations: int, errors: np.array, features_matrix: np.array, lambda1: float, lambda2: float, coefficients: np.array):
        
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors) + 2 * lambda1 * coefficients + lambda2 * np.sign(coefficients)
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)

        return gradient_coefficients, gradient_intercept
    
    #?____________________________________________________________________________________ #
    def gradient_descent(self, learning_rate: float, epochs: int, features_matrix: np.array, target_vector: np.array, lambda1: float, lambda2: float):
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
            learningRate = com.adapt_learning_rate(learningRate, loss, last_loss)
            early_stop = com.early_stopping(loss, last_loss)
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
    def fit(self, X_train, y_train):
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(self.params['learning_rate'], self.params['epochs'], X, y, self.params['lambda1'], self.params['lambda2'])
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        self.statistics, self.residuals = com.get_regression_stats(train_predictions, X, y, self.coefficients)
        
        return self.statistics
        
    #?____________________________________________________________________________________ #
    def predict(self, X_test):
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        
        return predictions
    
