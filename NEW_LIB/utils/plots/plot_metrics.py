from ..metrics import classification_metrics as cl
from ..metrics import financial_metrics as fm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



#! ==================================================================================== #
#! ============================ Classification Metrics ================================ #
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
    metrics = cl.generate_classification_metrics(predictions, labels, classes)

    # ======= II. Create confusion matrix and classification report =======
    confusion_matrix = cl.get_confusion_matrix(predictions, labels, classes)
    classification_report = cl.generate_classification_report(predictions, labels, classes)

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
#! ================================ Financial Metrics  ================================ #