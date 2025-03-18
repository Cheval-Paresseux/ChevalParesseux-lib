import numpy as np
from scipy.stats import t
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt


#! ==================================================================================== #
#! =============================== Auxiliary Functions ================================ #
def adapt_learning_rate(learning_rate: float, loss: float, last_loss: float):
    new_rate = learning_rate
    if loss > last_loss:
        new_rate /= 2
    else:
        new_rate *= 1.05
    
    return new_rate

#*____________________________________________________________________________________ #
def early_stopping(loss: float, last_loss: float):
    # ======= I. Check the loss diference =======
    if last_loss == np.inf:
        return False
    
    loss_diff = np.abs(loss - last_loss)
    early_stop = False
    
    # ======= II. Check if the loss difference is small enough =======
    if loss_diff < 1e-5:
        early_stop = True
    
    return early_stop



#! ==================================================================================== #
#! =================================== Base Models ==================================== #
class LinearRegression(ABC):

    def __init__(self):
        # --- Data Fitted ---
        self.X_train = None
        self.y_train = None
        
        self.X_test = None
        self.predictions = None
        
        # --- Model Parameters ---
        self.coefficients = None
        self.intercept = None
        
        # --- Model Statistics ---
        self.statistics = None
        self.residuals = None
        self.loss_history = None
        
    # ------------------------------- Auxiliary Functions -------------------------------- #
    def process_data(self, X_train, y_train):
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(X_train).reshape(-1, 1) if len(np.array(X_train).shape) == 1 else np.array(X_train)
        y = np.array(y_train)
        
        # ======= II. Store training data =======
        self.X_train = X
        self.y_train = y
        
        return X, y

    # ____________________________________________________________________________________ #
    @abstractmethod
    def gradient_descent(self, learning_rate: float, epochs: int, features_matrix: np.array, target_vector: np.array):
        pass

    # -------------------------------- Callable Functions -------------------------------- #
    def fit(self, X_train, y_train, learning_rate=0.001, epochs=1000):
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Perform Gradient Descent to extract the coefficients =======
        coefficients, intercept = self.gradient_descent(learning_rate, epochs, X, y)
        self.coefficients = coefficients
        self.intercept = intercept
        
    # ____________________________________________________________________________________ #
    def predict(self, X_test):
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        self.predictions = predictions
        
        return predictions
    
    # --------------------------------- Model Statistics --------------------------------- #
    def get_statistics(self):
        # ======= I. Extract the residuals =======
        predictions = self.predict(self.X_train)
        residuals = self.y_train - predictions
        
        # ======= II. Compute the residuals descriptive statistics =======
        nb_observations, nb_features = self.X_train.shape
        
        variance = np.sum(residuals**2) / (nb_observations - nb_features)
        mean = np.mean(residuals)
        median = np.median(residuals)

        # ======= III. Compute the R-squared =======
        SST = np.sum((self.y_train - np.mean(self.y_train))**2)
        SSR = np.sum((predictions - np.mean(self.y_train))**2)
        R_squared = SSR / SST
        
        # ======= IV. Compute the t-stats and p-values =======
        var_covar_matrix = variance * np.linalg.inv(self.X_train.T @ self.X_train)
        se_coefficients = np.sqrt(np.diag(var_covar_matrix))
        t_stats = self.coefficients / se_coefficients
        
        degrees_freedom = nb_observations - nb_features
        p_values = [2 * (1 - t.cdf(np.abs(t_stat), degrees_freedom)) for t_stat in t_stats]
        
        # ======= V. Store the statistics =======
        statistics = {
            "Variance": variance,
            "Mean": mean,
            "Median": median,
            "R_squared": R_squared,
            "T_stats": t_stats,
            "P_values": p_values
        }
        
        self.statistics = statistics
        self.residuals = residuals

        return statistics, residuals
    
    # -------------------------------- Model Visualization ------------------------------- #
    def plot_loss_history(self):
        
        plt.figure(figsize=(17, 5))
        plt.plot(self.loss_history, color="blue", linewidth=2)
        plt.title("Loss History", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.grid(True)
        plt.show()
        
        return None

    # ____________________________________________________________________________________ #
    def plot_fitted_line(self, feature_index=0):
        # ======= I. Extract the feature values =======
        X = self.X_train[:, feature_index]
        X = np.array(X).reshape(-1, 1)
        
        # ======= II. Compute the feature range =======
        min_feature = X.min()
        max_feature = X.max()
        feature_range = max_feature - min_feature
        rate = 0.001 * feature_range

        # ======= III. Create a range of feature values for plotting =======
        plotting_range = np.arange(min_feature - rate, max_feature + rate, rate)

        # ======= IV. Compute the fitted line =======
        fitted_line = self.intercept + plotting_range * self.coefficients[feature_index]

        # ======= V. Plot the fitted line =======
        plt.figure(figsize=(17, 5))
        plt.scatter(X, self.y_train, color="blue", label="Data Points")
        plt.plot(plotting_range, fitted_line, color="red", linewidth=2, label="Fitted Line")
        plt.title("Fitted Line", fontsize=16)
        plt.xlabel("Feature", fontsize=14)
        plt.ylabel("Target", fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()

        return None

    # ____________________________________________________________________________________ #
    def plot_residuals(self):
        # ======= I. Extract the residuals =======
        predictions = self.predict(self.X_train)
        residuals = self.y_train - predictions
        
        # ======= II. Compute the residuals descriptive statistics =======
        nb_observations, nb_features = self.X_train.shape
        
        variance = np.sum(residuals**2) / (nb_observations - nb_features)
        mean = np.mean(residuals)
        median = np.median(residuals)
        
        # ======= III. Plot the residuals =======
        plt.figure(figsize=(17, 5))
        plt.scatter(predictions, residuals, color="black")
        plt.axhline(y=mean, color="blue", linestyle="--")
        plt.axhline(y=median, color="pink", linestyle="--")
        plt.axhline(y=mean + np.sqrt(variance), color="green", linestyle="--")
        plt.axhline(y=mean - np.sqrt(variance), color="green", linestyle="--")
        plt.axhline(y=mean + 2 * np.sqrt(variance), color="red", linestyle="--")
        plt.axhline(y=mean - 2 * np.sqrt(variance), color="red", linestyle="--")
        plt.title("Residuals", fontsize=16)
        plt.xlabel("Predictions", fontsize=14)
        plt.ylabel("Residuals", fontsize=14)
        plt.grid(True)
        plt.show()
        
        return residuals
        


#! ==================================================================================== #
#! ================================ Regression Models ================================= #
class OLSRegression(LinearRegression):

    def __init__(self):
        super().__init__()
    
    def gradient_descent(self, learning_rate, epochs, features_matrix, target_vector):
         # Add a column of ones to X to account for the intercept
        X_with_intercept = np.c_[np.ones((features_matrix.shape[0], 1)), features_matrix]

        # Calculate the coefficients using the normal equation
        coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ target_vector

        # Extract the intercept and coefficients
        intercept = coefficients[0]
        coefficients = coefficients[1:]

        return coefficients, intercept

#*____________________________________________________________________________________ #
class MSERegression(LinearRegression):
    
    def __init__(self):
        super().__init__()

    # ____________________________________________________________________________________ #
    def MSE_gradient(self, nb_observations: int, errors: np.array, features_matrix: np.array):
        
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors)
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)
        
        return gradient_coefficients, gradient_intercept

    # ____________________________________________________________________________________ #
    def gradient_descent(self, learning_rate: float, epochs: int, features_matrix: np.array, target_vector: np.array):
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

#*____________________________________________________________________________________ #
class RidgeRegression(LinearRegression):
    
    def __init__(self, lambda_: float = 0.1):
        super().__init__()
        self.lambda_ = lambda_

    # ____________________________________________________________________________________ #
    def ridge_gradient(self, nb_observations: int, errors: np.array, features_matrix: np.array, lambda_: float, coefficients: np.array):
        
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors) + 2 * lambda_ * coefficients
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)

        return gradient_coefficients, gradient_intercept

    # ____________________________________________________________________________________ #
    def gradient_descent(self, learning_rate: float, epochs: int, features_matrix: np.array, target_vector: np.array):
        # ======= I. Initialize coefficients and intercept to 0 =======
        learningRate = learning_rate
        nb_observations, nb_features = features_matrix.shape
        lambda_ = self.lambda_
        
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

#*____________________________________________________________________________________ #
class LassoRegression(LinearRegression):
        
    def __init__(self, lambda_: float = 0.1):
        super().__init__()
        self.lambda_ = lambda_

    # ____________________________________________________________________________________ #
    def lasso_gradient(self, nb_observations: int, errors: np.array, features_matrix: np.array, lambda_: float, coefficients: np.array):
        
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors) + lambda_ * np.sign(coefficients)
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)

        return gradient_coefficients, gradient_intercept

    # ____________________________________________________________________________________ #
    def gradient_descent(self, learning_rate: float, epochs: int, features_matrix: np.array, target_vector: np.array):
        # ======= I. Initialize coefficients and intercept to 0 =======
        learningRate = learning_rate
        nb_observations, nb_features = features_matrix.shape
        lambda_ = self.lambda_
        
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

#*____________________________________________________________________________________ #
class ElasticNetRegression(LinearRegression):
        
    def __init__(self, lambda1: float = 0.1, lambda2: float = 0.1):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    # ____________________________________________________________________________________ #
    def elastic_net_gradient(self, nb_observations: int, errors: np.array, features_matrix: np.array, lambda1: float, lambda2: float, coefficients: np.array):
        
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors) + 2 * lambda1 * coefficients + lambda2 * np.sign(coefficients)
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)

        return gradient_coefficients, gradient_intercept

    # ____________________________________________________________________________________ #
    def gradient_descent(self, learning_rate: float, epochs: int, features_matrix: np.array, target_vector: np.array, lambda1: float = 1, lambda2: float = 1):
        # ======= I. Initialize coefficients and intercept to 0 =======
        learningRate = learning_rate
        nb_observations, nb_features = features_matrix.shape
        lambda1 = self.lambda1
        lambda2 = self.lambda2
        
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
            gradient_coefficients, gradient_intercept = self.elastic_net_gradient(nb_observations, errors, features_matrix, lambda1, lambda2, coefficients)

            # II.5 Update coefficients and intercept
            coefficients += learningRate * gradient_coefficients
            intercept += learningRate * gradient_intercept
        
        self.loss_history = loss_history

        return coefficients, intercept

