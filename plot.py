import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#*____________________________________________________________________________________ #
def generate_series_report(
    series: pd.Series
) -> dict:
    """
    Generate a report with basic information and descriptive statistics for a given Series.
    
    Parameters:
        - series (pd.Series): Input data series.
    
    Returns:
        - report (dict): Dictionary of basic and statistical properties.
    """
    # ======= I. Extract Basic Information =======
    data_type = series.dtype
    
    missing_values = series.isnull().sum()
    unique_values = series.nunique()
    zero_values = (series == 0).sum()
    negative_values = (series < 0).sum()
    positive_values = (series > 0).sum()

    # ======= II. Extract Descriptive Statistics =======
    mean = series.mean()
    median = series.median()
    min_val = series.min()
    max_val = series.max()
    std = series.std()
    skewness = series.skew()
    kurtosis = series.kurtosis()
    
    # ======= III. Store Basic Information =======
    report = {
        "data_type": data_type,
        "missing_values": missing_values,
        "unique_values": unique_values,
        "zero_values": zero_values,
        "negative_values": negative_values,
        "positive_values": positive_values,
        "mean": mean,
        "median": median,
        "min": min_val,
        "max": max_val,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis
    }
    
    return report

#*____________________________________________________________________________________ #
def plot_series_distribution(
    series: pd.Series, 
    figsize: tuple = (17, 5),
    title: str = 'Series Distribution',
) -> None:
    """
    Visualize the distribution and key statistics of a numerical Series.

    Parameters:
        - series (pd.Series): The Series to analyze.
        - title (str): Title of the plot.
    
    Returns:
        - None
    """
    # ======= I. Extract Report =======
    report = generate_series_report(series)
    mean = report['mean']
    median = report['median']
    min_val = report['min']
    max_val = report['max']
    std = report['std']
    skewness = report['skewness']
    kurtosis = report['kurtosis']

    # ======= II. Set up the figure for visualization =======
    plt.figure(figsize=figsize)
    
    # II.1 Plot histogram and KDE for the series
    sns.histplot(series, kde=True, bins=30, color="skyblue", stat="density", linewidth=0)
    
    # II.2 Plot first moments values
    plt.axvline(mean, color='orange', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    plt.axvline(min_val, color='red', linestyle='dashed', linewidth=2, label=f'Min: {min_val:.2f}')
    plt.axvline(max_val, color='blue', linestyle='dashed', linewidth=2, label=f'Max: {max_val:.2f}')
    plt.axvspan(mean - std, mean + std, color='yellow', alpha=0.3, label='±1 Std Dev')

    # II.3 Add skewness and kurtosis text box
    textstr = '\n'.join((f'Skewness: {skewness:.2f}', f'Kurtosis: {kurtosis:.2f}'))
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # II.4 Plot Normal Distribution for comparison
    x_vals = np.linspace(series.min(), series.max(), 200)
    normal_pdf = stats.norm.pdf(x_vals, loc=mean, scale=std)
    plt.plot(x_vals, normal_pdf, color='black', linestyle='--', linewidth=2, label='Normal PDF')

    # II.5 Final touches for the plot
    plt.title(title)
    plt.xlabel(f'Series Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

#*____________________________________________________________________________________ #
def plot_QQ(
    series: np.array,
    figsize: tuple = (17, 5),
    title: str = 'QQ Plot',
) -> None:
    """
    Generates a QQ plot to visually inspect normality of a series.
    
    Parameters:
        - series (np.array): The series to inspect.
        - title (str): Title of the plot.
    
    Returns:
        - None
    """
    # ======= I. Sort residuals =======
    sorted_residuals = np.sort(series)
    
    # ======= II. Generate theoretical quantiles =======
    quantiles = np.percentile(np.random.normal(0, 1, 10000), np.linspace(0, 100, len(sorted_residuals)))
    
    # ======= III. Create QQ plot =======
    plt.figure(figsize=figsize)
    plt.scatter(quantiles, sorted_residuals)
    plt.plot([min(quantiles), max(quantiles)], [min(sorted_residuals), max(sorted_residuals)], color='r', linestyle='--')
    plt.title(title)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.show()