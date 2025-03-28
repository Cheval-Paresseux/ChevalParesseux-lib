import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

#! ==================================================================================== #
#! ============================== Feature description ================================= #
def feature_data(feature_series: pd.Series):
    # ======= I. Extract Basic Information =======
    data_type = feature_series.dtype
    
    missing_values = feature_series.isnull().sum()
    unique_values = feature_series.nunique()
    zero_values = (feature_series == 0).sum()
    negative_values = (feature_series < 0).sum()
    positive_values = (feature_series > 0).sum()
    
    # ======= II. Visualizing Basic Information =======
    print(f"Data Type: {data_type}")
    print(f"Missing Values: {missing_values}, Unique Values: {unique_values}")
    print(f"Zero Values: {zero_values}, Negative Values: {negative_values}, Positive Values: {positive_values}")
    
    # ======= III. Store Basic Information =======
    basic_info = {
        "Data Type": data_type,
        "Missing Values": missing_values,
        "Unique Values": unique_values,
        "Zero Values": zero_values,
        "Negative Values": negative_values,
        "Positive Values": positive_values
    }
    
    return basic_info

#*____________________________________________________________________________________ #
def feature_distribution(feature_series: pd.Series, feature_name: str = None):
    # ======= O. Feature name =======
    if feature_name is None:
        feature_name = "Feature"
    
    # ======= I. Extract Descriptive Statistics =======
    mean = feature_series.mean()
    median = feature_series.median()
    min_val = feature_series.min()
    max_val = feature_series.max()
    std_dev = feature_series.std()
    skewness = feature_series.skew()
    kurtosis = feature_series.kurtosis()
    
    # ======= II. Store Descriptive Statistics =======
    descriptive_df = pd.DataFrame({"Mean": [mean], "Median": [median], "Min": [min_val], "Max": [max_val], "Std. Dev": [std_dev], "Skewness": [skewness], "Kurtosis": [kurtosis]}, index=[feature_name])
    
    # ======= III. Visualizing Descriptive Statistics =======
    plt.figure(figsize=(17, 5))
    sns.histplot(feature_series, kde=True, bins=30, color="skyblue", stat="density", linewidth=0, label=f"{feature_name} Distribution")

    plt.axvline(mean, color='orange', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    plt.axvline(min_val, color='red', linestyle='dashed', linewidth=2, label=f'Min: {min_val:.2f}')
    plt.axvline(max_val, color='blue', linestyle='dashed', linewidth=2, label=f'Max: {max_val:.2f}')
    plt.axvspan(mean - std_dev, mean + std_dev, color='yellow', alpha=0.3, label='±1 Std Dev')

    plt.title(f'{feature_name} Distribution with Key Statistics')
    plt.xlabel(f'{feature_name} Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return descriptive_df

#! ==================================================================================== #
#! ============================== Feature Analysis ==================================== #
def feature_plot(feature_series: pd.Series, label_series: pd.Series, feature_name: str = None):
    # ======= I. Visualization of the feature against labels =======
    plt.figure(figsize=(17, 5))
    plt.plot(label_series.index, feature_series, label=feature_name, linewidth=2)

    for i, label in label_series.items():
        if label == 1:
            plt.scatter(i, 0, color='green', label='Reg: Upward Movement', s=10, zorder=5)
        elif label == 0:
            plt.scatter(i, 0, color='black', label='Reg: Neutral Movement', s=10, zorder=5)
        elif label == -1:
            plt.scatter(i, 0, color='red', label='Reg: Downward Movement', s=10, zorder=5)

    plt.title(f'Feature {feature_name} against Labels')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(False)
    plt.show()
    
    label_feature_df = pd.DataFrame({'label': label_series, feature_name: feature_series})
    plt.figure(figsize=(17, 5))
    sns.boxplot(x='label', y=feature_name, data=label_feature_df)
    plt.title(f'Boxplot of {feature_name} by Label')
    plt.xlabel('Labels')
    plt.ylabel(f'{feature_name} Values')
    plt.show()

#*____________________________________________________________________________________ #
def features_correlation(label_feature_df: pd.DataFrame):
    # ======= I. Correlation between the feature and the labels =======
    light_green = (0, 0.7, 0.3, 0.2)
    colors = [(0, 'green'), (0.5, light_green), (0.99, 'green'), (1, 'grey')]
    n_bins = 1000
    cmap_name = 'green_white'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    corr_matrix = label_feature_df.corr()
    plt.figure(figsize=(17, 3))
    sns.heatmap(corr_matrix, annot=True, cmap=cm, vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

