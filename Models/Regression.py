import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# ==================================================================================== #
# ==================================== Base Model ==================================== #
class BaseModel(ABC):

    @abstractmethod
    def fit(self, X_train, y_train):
        """Fit the model to the data."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Make predictions using the model."""
        pass

# ==================================================================================== #
# ================================ Regression Models ================================= #
class LinearRegression(BaseModel):

    def __init__(self):
        # --- Data Fitted ---
        self.X_train = None
        self.y_train = None
        
        self.X_test = None
        
        # --- Model Parameters ---
        self.coefficients = None
        self.intercept = None
        
        # --- Model Statistics ---
        self.T_stats = None
        self.loss_history = None
        
    # ------------------------------- Auxiliary Functions ------------------------------- #
    def process_data(self, X, y):
        pass
    
    # ____________________________________________________________________________________ #
    def adapt_learning_rate(self, learning_rate: float, loss: float, last_loss: float):
        new_rate = learning_rate
        if loss > last_loss:
            new_rate /= 2
        else:
            new_rate *= 1.05
        
        return new_rate
    
    # ____________________________________________________________________________________ #
    def early_stopping(self, loss: float, last_loss: float):
        # ======= I. Check the loss diference =======
        if last_loss == np.inf:
            return False
        
        loss_diff = np.abs(loss - last_loss)
        early_stop = False
        
        # ======= II. Check if the loss difference is small enough =======
        if loss_diff < 1e-10:
            early_stop = True
        
        return early_stop
    
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
            learningRate = self.adapt_learning_rate(learningRate, loss, last_loss)
            early_stop = self.early_stopping(loss, last_loss)
            if early_stop:
                break
            last_loss = loss
            
            # II.4 Compute the gradient of the loss function (MSE)
            gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors)
            gradient_intercept = (-2 / nb_observations) * np.sum(errors) 

            # II.5 Update coefficients and intercept
            coefficients += learningRate * gradient_coefficients
            intercept += learningRate * gradient_intercept
        
        self.loss_history = loss_history

        return coefficients, intercept

    # ------------------------------- Callable Functions ------------------------------- #
    def fit(self, X_train, y_train, learning_rate=0.1, epochs=1000):
        # ======= I. Convert X and y to numpy arrays =======
        X = np.array(X_train).reshape(-1, 1) if len(np.array(X_train).shape) == 1 else np.array(X_train)
        y = np.array(y_train)

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
        
        return predictions
    
    # ------------------------------- Model Statistics ------------------------------- #
    def plot_loss_history(self):
        
        plt.figure(figsize=(17, 5))
        plt.plot(self.loss_history, color="blue", linewidth=2)
        plt.title("Loss History", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.grid(True)
        plt.show()
        
        return None
