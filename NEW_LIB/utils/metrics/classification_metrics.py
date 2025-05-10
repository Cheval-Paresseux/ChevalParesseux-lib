import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



#! ==================================================================================== #
#! ============================ Classification Metrics ================================ #
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

#*____________________________________________________________________________________ #
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


