from ..metrics import classification_metrics as cl
from ..metrics import financial_metrics as fi

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
import seaborn as sns



#! ==================================================================================== #
#! ============================ Classification Metrics ================================ #
def plot_classification_metrics(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> None:
    """
    Plot classification metrics including accuracy, precision, recall, and F1 score.
    
    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels. Default is [-1, 0, 1].
    
    Returns:
        - None
    """
    # ======= I. Compute metrics =======
    metrics = cl.generate_classification_metrics(predictions, labels, classes)

    # ======= II. Visualize metrics =======
    plt.figure(figsize=(17, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="coolwarm", hue=list(metrics.keys()), legend=False)
    plt.title("Overall Classification Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

#*____________________________________________________________________________________ #
def plot_classification_confusion(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> None:
    """
    Plot confusion matrix for classification predictions.
    
    Parameters:
        - predictions (pd.Series): Series of predicted labels.
        - labels (pd.Series): Series of true labels.
        - classes (list): List of class labels. Default is [-1, 0, 1].
    
    Returns:
        - None
    """
    # ======= I. Create confusion matrix =======
    confusion_matrix = cl.get_confusion_matrix(predictions, labels, classes)

    # ======= II. Confusion Matrix Heatmap ====
    conf_matrix_df = pd.DataFrame(confusion_matrix).T  
    plt.figure(figsize=(17, 4))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

#*____________________________________________________________________________________ #
def plot_classification_report(
    predictions: pd.Series, 
    labels: pd.Series, 
    classes: list = [-1, 0, 1]
) -> None:
    # ======= I. Compute Classification Report =======
    classification_report = cl.generate_classification_report(predictions, labels, classes)

    # ======= II. Classification Report Table ====
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

#! Following functions are to be tried and tested
#*____________________________________________________________________________________ #
def plot_roc_curve(
    y_true, 
    y_scores, 
    pos_label=1
) -> None:
    """
    Plot ROC curve and calculate AUC.
    
    Parameters:
        - y_true: True binary labels (1 or 0).
        - y_scores: Target scores (probabilities).
        - pos_label: The label of the positive class.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

#*____________________________________________________________________________________ #
def plot_precision_recall_curve(
    y_true, 
    y_scores, 
    pos_label=1
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
    ap = average_precision_score(y_true, y_scores, pos_label=pos_label)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    
#*____________________________________________________________________________________ #
def plot_calibration_curve(
    y_true, 
    y_prob, 
    n_bins=10
) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

#*____________________________________________________________________________________ #
def plot_probability_histogram(
    y_true, 
    y_prob, 
    bins=20
) -> None:
    plt.figure(figsize=(10, 5))
    for label in np.unique(y_true):
        sns.histplot(y_prob[y_true == label], bins=bins, kde=False, label=f"Class {label}", element="step", fill=True, alpha=0.5)
    plt.title("Histogram of Predicted Probabilities by True Class")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

#*____________________________________________________________________________________ #
def plot_prediction_errors(
    y_true, 
    y_pred, 
    y_prob=None
) -> None:
    errors = y_true != y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(y_prob[errors] if y_prob is not None else y_pred[errors], bins=20, color='red', label='Errors')
    plt.title("Distribution of Misclassified Predictions")
    plt.xlabel("Confidence or Predicted Class")
    plt.legend()
    plt.grid(True)
    plt.show()



#! ==================================================================================== #
#! ================================ Financial Metrics  ================================ #
def plot_quick_backtest(
    series: pd.Series,
    signal: pd.Series,
    frequence: str = 'daily',
    title: str = 'Quick Backtest',
    figsize: tuple = (17, 7),
):
    """
    Quick backtest of the signal on the series.

    Parameters:
        - series (pd.Series): Series of returns (e.g., daily or intraday).
        - signal (pd.Series): Series of signals (1 for buy, -1 for sell, 0 for hold).
        - frequence (str): Frequency of the return series. Must be one of:
        - title (str): Title of the plot.
        - figsize (tuple): Size of the figure.
    
    Returns:
        - None
    """
    # ======= I. Compute the returns =======
    returns = series.pct_change().shift(-1)
    signal_returns = returns * signal

    underlying_cum_returns = (1 + returns).cumprod()
    signal_cum_returns = (1 + signal_returns).cumprod()

    # ======= II. Plot the Cumulative Returns series =======
    plt.figure(figsize=figsize)
    plt.plot(signal_cum_returns, label='Signal Cumulative Returns')
    plt.plot(underlying_cum_returns, label=' Underlying Cumulative Returns')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

    # ======= III. Plot Performance =======
    plot_financial_performance(returns_series=signal_returns, market_returns=returns, risk_free_rate=0.0, frequence=frequence)

#*____________________________________________________________________________________ #
def plot_financial_distribution(
    returns_series: pd.Series,
    frequence: str = "daily"
) -> None:
    """
    Plot distribution metrics for a return series at a given frequency.
    
    Parameters:
        - returns_series (pd.Series): Series of returns (e.g., daily or intraday).
        - frequence (str): Frequency of the return series. Must be one of:
                           'daily', '5m', or '1m'.
                           Default is 'daily'.
    """
    # ======= I. Compute metrics =======
    metrics = fi.get_distribution(returns_series, frequence)

    # ======= II. Visualize metrics =======
    plt.figure(figsize=(17, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="coolwarm", hue=list(metrics.keys()), legend=False)
    plt.title("Overall Classification Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
    
#*____________________________________________________________________________________ #
def plot_financial_risks(
    returns_series: pd.Series,
) -> None:
    """
    Plot key downside risk metrics for a return series.
    
    Parameters:
        - returns_series (pd.Series): Series of returns (e.g., daily or intraday).    
    """
    # ======= I. Compute metrics =======
    metrics = fi.get_risk_measures(returns_series)

    # ======= II. Visualize metrics =======
    plt.figure(figsize=(17, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="coolwarm", hue=list(metrics.keys()), legend=False)
    plt.title("Overall Classification Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

#*____________________________________________________________________________________ #
def plot_financial_sensitivity(
    returns_series: pd.Series,
    market_returns: pd.Series,
    frequence: str = "daily"
) -> None:
    """
    Plot market sensitivity metrics for a return series.
    
    Parameters:
        - returns_series (pd.Series): Series of returns (e.g., daily or intraday).
        - market_returns (pd.Series): Series of market returns.
        - frequence (str): Frequency of the return series. Must be one of:
                           'daily', '5m', or '1m'.
                           Default is 'daily'.
    """
    # ======= I. Compute metrics =======
    metrics = fi.get_market_sensitivity(returns_series, market_returns, frequence)

    # ======= II. Visualize metrics =======
    plt.figure(figsize=(17, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="coolwarm", hue=list(metrics.keys()), legend=False)
    plt.title("Overall Classification Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

#*____________________________________________________________________________________ #
def plot_financial_performance(
    returns_series: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float,
    frequence: str = "daily"
) -> None:
    """
    Plot overall performance metrics for a return series.
    
    Parameters:
        - returns_series (pd.Series): Series of returns (e.g., daily or intraday).
        - market_returns (pd.Series): Series of market returns.
        - risk_free_rate (float): Risk-free rate.
        - frequence (str): Frequency of the return series. Must be one of:
                           'daily', '5m', or '1m'.
                           Default is 'daily'.
    """
    # ======= I. Compute metrics =======
    metrics = fi.get_performance_measures(returns_series, market_returns, risk_free_rate, frequence)

    # ======= II. Visualize metrics =======
    plt.figure(figsize=(17, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="coolwarm", hue=list(metrics.keys()), legend=False)
    plt.title("Overall Financial Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

