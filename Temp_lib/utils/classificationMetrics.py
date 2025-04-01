import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#! ==================================================================================== #
#! =============================== Main Visualizer ==================================== #
def classification_metrics(predictions: pd.Series, labels: pd.Series, classes: list = [-1, 0, 1]):
    # ======= I. Compute metrics =======
    metrics = {
        "Accuracy": get_accuracy(predictions, labels),
        "Precision": get_precision(predictions, labels, classes),
        "Recall": get_recall(predictions, labels, classes),
        "F1 Score": get_f1_score(predictions, labels, classes),
        "Balanced Accuracy": get_balanced_accuracy(predictions, labels, classes),
        "MCC": get_MCC(predictions, labels, classes),
        "Cohen Kappa": get_cohen_kappa(predictions, labels, classes)
    }

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

#! ==================================================================================== #
#! =============================== Metrics Functions ================================== #
def get_accuracy(predictions: pd.Series, labels: pd.Series) -> float:
    
    # ======= I. Compute the number of accurate predictions =======
    correct_predictions = (predictions == labels).sum()

    # ======= II. Compute the Accuracy =======
    total_predictions = len(labels)
    accuracy = correct_predictions / total_predictions

    return accuracy

#*____________________________________________________________________________________ #
def get_precision(predictions: pd.Series, labels: pd.Series, classes: list = [-1, 0, 1]) -> float:
    
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
def get_recall(predictions: pd.Series, labels: pd.Series, classes: list = [-1, 0, 1]) -> float:
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
def get_f1_score(predictions: pd.Series, labels: pd.Series, classes: list = [-1, 0, 1]) -> float:
    
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
def get_confusion_matrix(predictions: pd.Series, labels: pd.Series, classes: list = [-1, 0, 1]) -> dict:
    
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
def get_classification_report(predictions: pd.Series, labels: pd.Series, classes: list = [-1, 0, 1]) -> dict:
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
def get_balanced_accuracy(predictions: pd.Series, labels: pd.Series, classes: list = [-1, 0, 1]) -> float:
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
def get_MCC(predictions: pd.Series, labels: pd.Series, classes: list = [-1, 0, 1]) -> float:
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
def get_cohen_kappa(predictions: pd.Series, labels: pd.Series, classes: list = [-1, 0, 1]) -> float:
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