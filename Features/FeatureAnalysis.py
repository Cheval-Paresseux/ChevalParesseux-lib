import auxiliary as aux

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================================================== #
# ============================== Feature description ================================= #
def feature_data(feature_series: pd.Series):
    # ======= I. Extract Basic Information =======
    feature_name = feature_series.name
    data_type = feature_series.dtype
    
    missing_values = feature_series.isnull().sum()
    unique_values = feature_series.nunique()
    zero_values = (feature_series == 0).sum()
    negative_values = (feature_series < 0).sum()
    positive_values = (feature_series > 0).sum()
    
    # ======= II. Visualizing Basic Information =======
    print(f"Feature Name: {feature_name}, Data Type: {data_type}")
    print(f"Missing Values: {missing_values}, Unique Values: {unique_values}")
    print(f"Zero Values: {zero_values}, Negative Values: {negative_values}, Positive Values: {positive_values}")
    
    # ======= III. Store Basic Information =======
    basic_info = {
        "Feature Name": feature_name,
        "Data Type": data_type,
        "Missing Values": missing_values,
        "Unique Values": unique_values,
        "Zero Values": zero_values,
        "Negative Values": negative_values,
        "Positive Values": positive_values
    }
    
    return basic_info

# ____________________________________________________________________________________ #
def feature_distribution(feature_series: pd.Series):
    # ======= O. Feature name =======
    feature_name = feature_series.name
    
    # ======= I. Extract Descriptive Statistics =======
    mean = feature_series.mean()
    median = feature_series.median()
    min_val = feature_series.min()
    max_val = feature_series.max()
    std_dev = feature_series.std()
    mode = feature_series.mode().values
    skewness = feature_series.skew()
    kurtosis = feature_series.kurtosis()
    
    # ======= II. Visualizing Descriptive Statistics =======
    plt.figure(figsize=(17, 5))
    sns.histplot(feature_series, kde=True, bins=30, color="skyblue", stat="density", linewidth=0, label=f"{feature_name} Distribution")

    plt.axvline(mean, color='orange', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    plt.axvline(min_val, color='red', linestyle='dashed', linewidth=2, label=f'Min: {min_val:.2f}')
    plt.axvline(max_val, color='blue', linestyle='dashed', linewidth=2, label=f'Max: {max_val:.2f}')
    plt.axvline(mode, color='purple', linestyle='dashed', linewidth=2, label=f'Mode: {mode[0]:.2f}')
    plt.axvspan(mean - std_dev, mean + std_dev, color='yellow', alpha=0.3, label='±1 Std Dev')

    plt.title(f'{feature_name} Distribution with Key Statistics')
    plt.xlabel(f'{feature_name} Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ======= III. Store Descriptive Statistics =======
    descriptive_df = pd.DataFrame({"Mean": mean, "Median": median, "Min": min_val, "Max": max_val, "Std. Dev": std_dev, "Mode": mode, "Skewness": skewness, "Kurtosis": kurtosis})
    
    return descriptive_df

# ____________________________________________________________________________________ #

    
    