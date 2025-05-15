import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#! ==================================================================================== #
#! ============================ Classification Metrics ================================ #
#?__________________________________ Main Functions ___________________________________ #
def plot_classification_metrics(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> None:
    """
    Plot classification metrics including accuracy, precision, recall, F1 score, confusion matrix, and classification report.
    
    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels. Default is [-1, 0, 1].
    
    Returns:
        - None
    """
    # ======= I. Compute metrics =======
    metrics = get_classification_metrics(predictions, labels, classes)

    # ======= II. Create confusion matrix and classification report =======
    confusion_matrix = get_confusion_matrix(predictions, labels, classes)
    classification_report = get_classification_report(predictions, labels, classes)

    # ======= III. Visualize metrics =======
    plt.figure(figsize=(17, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="coolwarm", hue=list(metrics.keys()), legend=False)
    plt.title("Overall Classification Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # ======= IV. Confusion Matrix Heatmap ====
    conf_matrix_df = pd.DataFrame(confusion_matrix).T  
    plt.figure(figsize=(17, 4))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # ======= V. Classification Report Table ====
    class_report_df = pd.DataFrame(classification_report).T
    plt.figure(figsize=(17, 3))
    ax = plt.subplot()
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=class_report_df.round(3).values,  
        colLabels=class_report_df.columns,
        rowLabels=class_report_df.index,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)  

    for key, cell in table._cells.items():
        if key[0] == 0:  # First row (column headers)
            cell.set_text_props(fontweight="bold")  # Bold text
            cell.set_facecolor("#dddddd")
            
    plt.title("Classification Report", fontsize=16, fontweight="bold", pad=15)
    plt.show()

#*____________________________________________________________________________________ #
def generate_classification_metrics(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> dict:
    """
    Compute and store classification metrics.
    
    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels. Default is [-1, 0, 1].
    
    Returns:
        - metrics (dict): Dictionary containing classification metrics.
    """
    metrics = {
        "accuracy": get_accuracy(predictions, labels),
        "precision": get_precision(predictions, labels, classes),
        "recall": get_recall(predictions, labels, classes),
        "f1_score": get_f1_score(predictions, labels, classes),
        "balanced_accuracy": get_balanced_accuracy(predictions, labels, classes),
        "mcc": get_MCC(predictions, labels, classes),
        "cohen_kappa": get_cohen_kappa(predictions, labels, classes)
    }
    
    return metrics

#*____________________________________________________________________________________ #
def generate_classification_report(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> dict:
    """
    Generate a classification report with precision, recall, and F1 score for each class.

    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels.

    Returns:
        - report (dict): Classification report by class.
    """
    # ======= I. Initialize the classification report =======
    report = {}

    # ======= II. Compute Precision, Recall, and F1 Score for each class =======
    for value in classes:
        true_positives = ((predictions == value) & (labels == value)).sum()
        false_positives = ((predictions == value) & (labels != value)).sum()
        false_negatives = ((predictions != value) & (labels == value)).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        report[value] = {"Precision": float(precision), "Recall": float(recall), "F1-score": float(f1_score)}
    
    return report


#?_____________________________ Metrics Functions ___________________________________ #
def get_accuracy(
    predictions: pd.Series, 
    labels: pd.Series
) -> float:
    """
    Compute the accuracy of predictions : Proportion of correct predictions.
    
    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
    
    Returns:
        - accuracy (float): Accuracy of predictions.
    """
    # ======= I. Compute the number of accurate predictions =======
    correct_predictions = (predictions == labels).sum()

    # ======= II. Compute the Accuracy =======
    total_predictions = len(labels)
    accuracy = correct_predictions / total_predictions

    return accuracy

#*____________________________________________________________________________________ #
def get_precision(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> float:
    """
    Compute the precision of predictions : Proportion of true positive predictions among all positive predictions.
    
    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels. Default is [-1, 0, 1].
    
    Returns:
        - precision (float): Precision of predictions.
    """
    # ======= I. Identify positive classes =======
    positive_classes = [value for value in classes if value > 0]

    # ======== II. Compute the number of True Positives and False Positives =======
    true_positives = ((predictions.isin(positive_classes)) & (labels.isin(positive_classes))).sum()
    false_positives = ((predictions.isin(positive_classes)) & (~labels.isin(positive_classes))).sum()

    # ======= III. Compute Precision =======
    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0
    
    return precision

#*____________________________________________________________________________________ #
def get_recall(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> float:
    """
    Compute the recall: proportion of true positive predictions among all actual positives.

    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels.

    Returns:
        - recall (float): Recall of predictions.
    """
    # ======= I. Identify positive classes =======
    positive_classes = [value for value in classes if value > 0]

    # ======== II. Compute the number of True Positives and False Negatives =======
    true_positives = ((predictions.isin(positive_classes)) & (labels.isin(positive_classes))).sum()
    false_negatives = ((~predictions.isin(positive_classes)) & (labels.isin(positive_classes))).sum()

    # ======= III. Compute Recall =======
    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0
    
    return recall

#*____________________________________________________________________________________ #
def get_f1_score(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> float:
    """
    Compute the F1 score: harmonic mean of precision and recall.

    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels.

    Returns:
        - f1_score (float): F1 score of predictions.
    """
    # ======= I. Compute Precision and Recall =======
    precision = get_precision(predictions, labels, classes)
    recall = get_recall(predictions, labels, classes)

    # ======= II. Compute F1 Score =======
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return f1_score

#*____________________________________________________________________________________ #
def get_confusion_matrix(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> dict:
    """
    Compute the confusion matrix as a nested dictionary.

    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels.

    Returns:
        - matrix (dict): Confusion matrix with actual vs predicted counts.
    """
    # ======= I. Initialize the confusion matrix =======
    # The confusion matrix is a dictionary where the keys are the actual classes and the values are dictionaries
    matrix = {c: {c_: 0 for c_ in classes} for c in classes}

    # ======= II. Fill the confusion matrix =======
    # For each prediction and actual label, increment the corresponding cell in the matrix
    # We assume that the predictions and labels are aligned and of the same length
    for prediction, label in zip(predictions, labels):
        matrix[label][prediction] += 1
    
    return matrix

#*____________________________________________________________________________________ #
def get_balanced_accuracy(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> float:
    """
    Compute balanced accuracy: average recall over all classes.

    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels.

    Returns:
        - balanced_accuracy (float): Balanced accuracy of predictions.
    """
    # ======= I. Initialize =======
    recall_per_class = []

    # ======= II. Compute Recall for each class =======
    for value in classes:
        true_positives = ((predictions == value) & (labels == value)).sum()
        false_negatives = ((predictions != value) & (labels == value)).sum()
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        recall_per_class.append(recall)
    
    # ======= III. Compute Balanced Accuracy =======
    balanced_accuracy = sum(recall_per_class) / len(classes)

    return balanced_accuracy

#*____________________________________________________________________________________ #
def get_MCC(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> float:
    """
    Compute the Matthews Correlation Coefficient (MCC).

    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels.

    Returns:
        - mcc (float): Matthews Correlation Coefficient.
    """
    # ======= I. Initialize =======
    nb_classes = len(labels)
    preds = predictions.value_counts().reindex(classes, fill_value=0)
    labls = labels.value_counts().reindex(classes, fill_value=0)
    
    sum_correct_predictions = sum((predictions == labels) & labels.isin(classes))
    P_k = sum(preds[c]**2 for c in classes)
    T_k = sum(labls[c]**2 for c in classes)

    # ======= II. Compute Matthews Correlation Coefficient =======
    denominator = np.sqrt((nb_classes**2 - P_k) * (nb_classes**2 - T_k))
    numerator = (nb_classes * sum_correct_predictions - sum(preds[c] * labls[c] for c in classes))
    mcc = numerator / denominator if denominator > 0 else 0.0

    return mcc

#*____________________________________________________________________________________ #
def get_cohen_kappa(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> float:
    """
    Compute Cohen's Kappa: agreement between predictions and labels adjusted for chance.

    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels.

    Returns:
        - cohen_kappa (float): Cohen's Kappa score.
    """
    # ======= I. Compute the Observed Agreement =======
    total = len(labels)
    observed_agreement = (predictions == labels).sum() / total
    
    # ======= II. Compute the Expected Agreement =======
    expected_agreement = sum(
        (predictions.value_counts(normalize=True).get(value, 0) * labels.value_counts(normalize=True).get(value, 0))
        for value in classes
    )

    # ======= III. Compute Cohen's Kappa =======
    if (1 - expected_agreement) > 0:
        cohen_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    else:
        cohen_kappa = 0.0

    return cohen_kappa



#! ==================================================================================== #
#! ================================ Financial Metrics  ================================ #
def get_distribution(
    returns_series: pd.Series, 
    frequence: str = "daily"
) -> dict:
    """
    Compute distribution statistics for a return series at a given frequency.

    Parameters:
        - returns_series (pd.Series): Series of returns (e.g., daily or intraday).
        - frequence (str): Frequency of the return series. Must be one of:
                           'daily', '5m', or '1m'.
                           Default is 'daily'.

    Returns:
        - distribution_stats (dict): Dictionary containing the following statistics:
            - expected_return: Annualized mean return.
            - volatility: Annualized standard deviation of returns.
            - downside_deviation: Annualized standard deviation of negative returns.
            - median_return: Annualized median return.
            - skew: Skewness of the return distribution.
            - kurtosis: Kurtosis of the return distribution.
    """
    # ======= I. Get the right frequence =======
    frequence_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequence = frequence_dict[frequence]
    
    # ======= II. Compute the statistics =======
    expected_return = returns_series.mean() * adjusted_frequence
    volatility = returns_series.std() * np.sqrt(adjusted_frequence)
    downside_deviation = returns_series[returns_series < 0].std() * np.sqrt(adjusted_frequence) if returns_series[returns_series < 0].sum() != 0 else 0
    median_return = returns_series.median() * adjusted_frequence
    skew = returns_series.skew()
    kurtosis = returns_series.kurtosis()
    
    # ======= III. Store the statistics =======
    distribution_stats = {
        "expected_return": expected_return,
        "volatility": volatility,
        "downside_deviation": downside_deviation,
        "median_return": median_return,
        "skew": skew,
        "kurtosis": kurtosis,
    }
    
    return distribution_stats

#*____________________________________________________________________________________ #
def get_risk_measures(
    returns_series: pd.Series
) -> dict:
    """
    Compute key downside risk metrics for a return series.

    Parameters:
        - returns_series (pd.Series): Series of periodic returns.

    Returns:
        - risk_stats (dict): Dictionary containing:
            - mean_drawdown: Average drawdown over the period.
            - maximum_drawdown: Maximum observed drawdown.
            - max_drawdown_duration: Maximum duration of drawdowns (in periods).
            - var_95: 5% Value at Risk (VaR).
            - cvar_95: Conditional VaR at 5% (Expected Shortfall).
    """
    # ======= I. Compute the Cumulative returns =======    
    cumulative_returns = (1 + returns_series).cumprod()
    
    # ======= II. Compute the statistics =======
    # ------ Maximum Drawdown and Duration
    running_max = cumulative_returns.cummax().replace(0, 1e-10)
    drawdown = (cumulative_returns / running_max) - 1
    drawdown_durations = (drawdown < 0).astype(int).groupby((drawdown == 0).cumsum()).cumsum()
    
    mean_drawdown = drawdown.mean()

    maximum_drawdown = drawdown.min()
    max_drawdown_duration = drawdown_durations.max()

    # ------ Value at Risk and Conditional Value at Risk
    var_95 = returns_series.quantile(0.05)
    cvar_95 = returns_series[returns_series <= var_95].mean()
    
    # ======= III. Store the statistics =======
    risk_stats = {
        "mean_drawdown": mean_drawdown,
        "maximum_drawdown": maximum_drawdown,
        "max_drawdown_duration": max_drawdown_duration,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }
    
    return risk_stats

#*____________________________________________________________________________________ #
def get_market_sensitivity(
    returns_series: pd.Series, 
    market_returns: pd.Series, 
    frequence: str = "daily"
) -> dict:
    """
    Estimate the sensitivity of a strategy or asset to market returns.

    Parameters:
        - returns_series (pd.Series): Asset or strategy return series.
        - market_returns (pd.Series): Benchmark or market return series.
        - frequence (str): Frequency of data ('daily', '5m', or '1m'). Default is 'daily'.

    Returns:
        - market_sensitivity_stats (dict): Dictionary containing:
            - beta: Market beta coefficient.
            - alpha: Jensen's alpha (annualized).
            - upside_capture: Average return ratio in positive market periods.
            - downside_capture: Average return ratio in negative market periods.
            - tracking_error: Annualized tracking error vs. the market.
    """
    # ======= I. Get the right frequence =======
    frequence_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequence = frequence_dict[frequence]
    
    # ======= II. Compute the statistics =======
    # ------ Beta and Alpha (Jensens's)
    beta = returns_series.cov(market_returns) / market_returns.var()
    alpha = returns_series.mean() * adjusted_frequence - beta * (market_returns.mean() * adjusted_frequence)
    
    # ------ Capture Ratios
    upside_capture = returns_series[market_returns > 0].mean() / market_returns[market_returns > 0].mean()
    downside_capture = returns_series[market_returns < 0].mean() / market_returns[market_returns < 0].mean()

    # ------ Tracking Error
    tracking_error = returns_series.sub(market_returns).std() * np.sqrt(adjusted_frequence)
    
    # ======= III. Store the statistics =======
    market_sensitivity_stats = {
        "beta": beta,
        "alpha": alpha,
        "upside_capture": upside_capture,
        "downside_capture": downside_capture,
        "tracking_error": tracking_error,
    }
    
    return market_sensitivity_stats

#*____________________________________________________________________________________ #
def get_performance_measures(
    returns_series: pd.Series, 
    market_returns: pd.Series, 
    risk_free_rate: float = 0.0, 
    frequence: str = "daily"
) -> tuple:
    """
    Compute classic performance ratios and supporting metrics for a strategy.

    Parameters:
        - returns_series (pd.Series): Strategy or asset return series.
        - market_returns (pd.Series): Benchmark return series.
        - risk_free_rate (float): Annualized risk-free rate. Default is 0.0.
        - frequence (str): Data frequency ('daily', '5m', or '1m'). Default is 'daily'.

    Returns:
        - performance_stats (dict): Dictionary containing:
            - sharpe_ratio
            - sortino_ratio
            - treynor_ratio
            - information_ratio
            - sterling_ratio
            - calmar_ratio
        - details (tuple): Tuple of three dictionaries:
            (distribution_stats, risk_stats, market_sensitivity_stats)
    """
    # ======= I. Get the right frequence =======
    frequence_dict = {"daily": 252, "5m": 19656, "1m": 98280}
    adjusted_frequence = frequence_dict[frequence]
    
    # ======= II. Extract Statistics =======
    distribution_stats = get_distribution(returns_series, frequence)
    expected_return = distribution_stats["expected_return"]
    volatility = distribution_stats["volatility"]
    downside_deviation = distribution_stats["downside_deviation"]
    
    risk_stats = get_risk_measures(returns_series)
    mean_drawdown = risk_stats["mean_drawdown"]
    maximum_drawdown = risk_stats["maximum_drawdown"]
    
    market_sensitivity_stats = get_market_sensitivity(returns_series, market_returns, frequence)
    beta = market_sensitivity_stats["beta"]
    tracking_error = market_sensitivity_stats["tracking_error"]
    
    # ======= III. Compute the ratios =======
    # ------ Sharpe, Sortino, Treynor, and Information Ratios
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility != 0 else 0
    sortino_ratio = expected_return / downside_deviation if downside_deviation != 0 else 0
    treynor_ratio = expected_return / beta if beta != 0 else 0
    information_ratio = (expected_return - market_returns.mean() * adjusted_frequence) / tracking_error if tracking_error != 0 else 0

    # ------ Sterling, and Calmar Ratios
    average_drawdown = abs(mean_drawdown) if mean_drawdown != 0 else 0
    sterling_ratio = (expected_return - risk_free_rate) / average_drawdown if average_drawdown != 0 else 0
    calmar_ratio = expected_return / abs(maximum_drawdown) if maximum_drawdown != 0 else 0
    
    # ======= IV. Store the statistics =======
    performance_stats = {
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "treynor_ratio": treynor_ratio,
        "information_ratio": information_ratio,
        "sterling_ratio": sterling_ratio,
        "calmar_ratio": calmar_ratio,
    }
    
    return performance_stats, distribution_stats, risk_stats, market_sensitivity_stats

