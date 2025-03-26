from ..Models import common as com

import numpy as np


#! ==================================================================================== #
#! ================================ Regression Models ================================= #
class OLSRegression(com.ML_Model):
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
    def normal_equation(self, features_matrix: np.array, target_vector: np.array):
        X_with_intercept = np.c_[np.ones((features_matrix.shape[0], 1)), features_matrix]

        # Compute coefficients using the normal equation
        XTX = X_with_intercept.T @ X_with_intercept
        XTy = X_with_intercept.T @ target_vector

        # Use pseudo-inverse instead of direct inversion (handles singular matrices)
        coefficients = np.linalg.pinv(XTX) @ XTy

        # Extract intercept and coefficients
        intercept = coefficients[0]
        coefficients = coefficients[1:]

        return coefficients, intercept
    
    #?_____________________________ User Functions _______________________________________ #
    def fit(self, X_train, y_train):
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.normal_equation(X, y)
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        self.statistics, self.residuals = com.get_regression_stats(train_predictions, X, y, self.coefficients)
        
        return train_predictions
        
    #?____________________________________________________________________________________ #
    def predict(self, X_test):
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        self.predictions = predictions
        
        return predictions
    
#*____________________________________________________________________________________ #
class MSERegression(com.ML_Model):
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
    def MSE_gradient(self, nb_observations: int, errors: np.array, features_matrix: np.array):
        
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors)
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)
        
        return gradient_coefficients, gradient_intercept
    
    #?____________________________________________________________________________________ #
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
            learningRate = com.adapt_learning_rate(learningRate, loss, last_loss)
            early_stop = com.early_stopping(loss, last_loss)
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
    def fit(self, X_train, y_train, learning_rate: float = 0.01, epochs: int = 1000):
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(learning_rate, epochs, X, y)
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        self.statistics, self.residuals = com.get_regression_stats(train_predictions, X, y, self.coefficients)
        
        return train_predictions
        
    #?____________________________________________________________________________________ #
    def predict(self, X_test):
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        self.predictions = predictions
        
        return predictions
    
#*____________________________________________________________________________________ #
class RidgeRegression(com.ML_Model):
    def __init__(self, lambda_: float = 0.1):
        # --- Data Fitted ---
        self.X_train = None
        self.y_train = None
        
        self.X_test = None
        self.predictions = None
        
        # --- Model Parameters ---
        self.coefficients = None
        self.intercept = None
        self.lambda_ = lambda_
        
        # --- Model Statistics ---
        self.statistics = None
        self.residuals = None
        self.loss_history = None
        
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
    def ridge_gradient(self, nb_observations: int, errors: np.array, features_matrix: np.array, lambda_: float, coefficients: np.array):
        
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors) + 2 * lambda_ * coefficients
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)

        return gradient_coefficients, gradient_intercept
    
    #?____________________________________________________________________________________ #
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
            learningRate = com.adapt_learning_rate(learningRate, loss, last_loss)
            early_stop = com.early_stopping(loss, last_loss)
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
    def fit(self, X_train, y_train, learning_rate: float = 0.01, epochs: int = 1000):
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(learning_rate, epochs, X, y)
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        self.statistics, self.residuals = com.get_regression_stats(train_predictions, X, y, self.coefficients)
        
        return train_predictions
        
    #?____________________________________________________________________________________ #
    def predict(self, X_test):
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        self.predictions = predictions
        
        return predictions
    
#*____________________________________________________________________________________ #
class LassoRegression(com.ML_Model):
    def __init__(self, lambda_: float = 0.1):
        # --- Data Fitted ---
        self.X_train = None
        self.y_train = None
        
        self.X_test = None
        self.predictions = None
        
        # --- Model Parameters ---
        self.coefficients = None
        self.intercept = None
        self.lambda_ = lambda_
        
        # --- Model Statistics ---
        self.statistics = None
        self.residuals = None
        self.loss_history = None
        
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
    def lasso_gradient(self, nb_observations: int, errors: np.array, features_matrix: np.array, lambda_: float, coefficients: np.array):
        
        gradient_coefficients = (-2 / nb_observations) * np.dot(features_matrix.T, errors) + lambda_ * np.sign(coefficients)
        gradient_intercept = (-2 / nb_observations) * np.sum(errors)

        return gradient_coefficients, gradient_intercept
    
    #?____________________________________________________________________________________ #
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
            learningRate = com.adapt_learning_rate(learningRate, loss, last_loss)
            early_stop = com.early_stopping(loss, last_loss)
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
    def fit(self, X_train, y_train, learning_rate: float = 0.01, epochs: int = 1000):
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(learning_rate, epochs, X, y)
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        self.statistics, self.residuals = com.get_regression_stats(train_predictions, X, y, self.coefficients)
        
        return train_predictions
        
    #?____________________________________________________________________________________ #
    def predict(self, X_test):
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        self.predictions = predictions
        
        return predictions
    
#*____________________________________________________________________________________ #
class ElasticNetRegression(com.ML_Model):
    def __init__(self, lambda1: float = 0.1, lambda2: float = 0.1):
        # --- Data Fitted ---
        self.X_train = None
        self.y_train = None
        
        self.X_test = None
        self.predictions = None
        
        # --- Model Parameters ---
        self.coefficients = None
        self.intercept = None
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # --- Model Statistics ---
        self.statistics = None
        self.residuals = None
        self.loss_history = None
        
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
    def fit(self, X_train, y_train, learning_rate: float = 0.01, epochs: int = 1000):
        # ======= I. Process Data =======
        X, y = self.process_data(X_train, y_train)
        
        # ======= II. Solve for coefficients =======
        self.coefficients, self.intercept = self.gradient_descent(learning_rate, epochs, X, y)
        
        # ======= III. Compute Training Predictions =======
        train_predictions = self.predict(X)

        # ======= IV. Compute Statistics =======
        self.statistics, self.residuals = com.get_regression_stats(train_predictions, X, y, self.coefficients)
        
        return train_predictions
        
    #?____________________________________________________________________________________ #
    def predict(self, X_test):
        # ======= I. Convert X to a numpy array =======
        X = np.array(X_test).reshape(-1, 1) if len(np.array(X_test).shape) == 1 else np.array(X_test)
        
        # ======= II. Make predictions =======
        predictions = self.intercept + np.dot(X, self.coefficients)
        self.predictions = predictions
        
        return predictions
    
