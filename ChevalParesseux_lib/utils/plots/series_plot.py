from ..measures import trending_measures as tm

import pandas as pd
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns



#! ==================================================================================== #
#! ============================== Series description ================================== #
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

    # II.3 Confidence intervals (as vertical lines)
    confidence_levels = [0.90, 0.95, 0.99]
    line_styles = ['solid', 'dashed', 'dotted']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    
    for level, style, color in zip(confidence_levels, line_styles, colors):
        z = stats.norm.ppf((1 + level) / 2)
        lower = mean - z * std
        upper = mean + z * std
        plt.axvline(lower, color=color, linestyle=style, linewidth=1.5, label=f'{int(level * 100)}% CI Lower')
        plt.axvline(upper, color=color, linestyle=style, linewidth=1.5, label=f'{int(level * 100)}% CI Upper')

    # II.4 Add skewness and kurtosis text box
    textstr = '\n'.join((f'Skewness: {skewness:.2f}', f'Kurtosis: {kurtosis:.2f}'))
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # II.5 Plot Normal Distribution for comparison
    x_vals = np.linspace(series.min(), series.max(), 200)
    normal_pdf = stats.norm.pdf(x_vals, loc=mean, scale=std)
    plt.plot(x_vals, normal_pdf, color='black', linestyle='--', linewidth=2, label='Normal PDF')

    # II.6 Final touches for the plot
    plt.title(title)
    plt.xlabel('Series Values')
    plt.ylabel('Density')
    plt.legend(
        bbox_to_anchor=(1.01, 1), 
        loc='upper left', 
        borderaxespad=0.,
        frameon=True
    )
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

#*____________________________________________________________________________________ #
def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: tuple = (17, 5),
    title: str = "Correlation Matrix",
    legend: bool = False,
):
    """
    Plot a correlation matrix using a custom colormap.
    
    Parameters:
        - df (pd.DataFrame): DataFrame containing the data to analyze.
        - title (str): Title of the plot.
        - figsize (tuple): Size of the figure.
        - legend (bool): Whether to show the colorbar legend.
    """
    # ======= I. Custom colormap =======
    light_green = (0, 0.7, 0.3, 0.2)
    colors = [(0, 'green'), (0.5, light_green), (0.99, 'green'), (1, 'grey')]
    cmap_name = 'green_white'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=1000)

    # ======= II. Compute and plot =======
    corr_matrix = df.corr()
    plt.figure(figsize=figsize)
    ax = sns.heatmap(corr_matrix, annot=True, cmap=cm, vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)

    # ======= III. Remove axis labels and ticks =======
    if legend:
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['-1', '0', '1'])
        cbar.ax.tick_params(labelsize=10)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.title(title)
    plt.tight_layout()
    plt.show()

#*____________________________________________________________________________________ #
def plot_acf(
    series: pd.Series, 
    lags: int = 20, 
    figsize: tuple = (17, 5),
    title: str = "Series"
) -> None:
    """
    Plot Autocorrelation (ACF) for a given series.
    
    Parameters:
        - series (pd.Series): Time series data.
        - lags (int): Number of lags to compute.
        - figsize (tuple): Size of the figure.
        - title (str): Title of the plot.
    
    Returns:
        - None
    """
    # ======= I. Check for constant series =======
    series = pd.Series(series).dropna()
    if series.nunique() <= 1:
        print("Series is constant — cannot compute ACF/PACF.")
        return
    
    # ======= II. Compute ACF values =======
    acf_vals = tm.get_autocorrelation(series, lags)
    N = len(series)
    confidence_interval = 1.96 / np.sqrt(N)

    # ======= III. Plot ACF =======
    plt.figure(figsize=figsize)
    plt.stem(range(1, len(acf_vals)+1), acf_vals, basefmt=" ")
    plt.axhline(confidence_interval, linestyle='--', color='red', alpha=0.5)
    plt.axhline(-confidence_interval, linestyle='--', color='red', alpha=0.5)
    plt.axhline(0, linestyle='--', color='gray')
    plt.title(title)
    plt.ylabel("ACF")
    plt.xlabel("Lag")
    plt.tight_layout()
    plt.show()

#*____________________________________________________________________________________ #
def plot_pacf(
    series: pd.Series, 
    lags: int = 20, 
    figsize: tuple = (17, 5),
    title: str = "Series"
) -> None:
    """
    Plot Partial Autocorrelation (PACF) for a given series.

    Parameters:
        - series (pd.Series): Time series data.
        - lags (int): Number of lags to compute.
        - figsize (tuple): Size of the figure.
        - title (str): Title of the plot.
    
    Returns:
        - None
    """
    # ======= I. Check for constant series =======
    series = pd.Series(series).dropna()
    if series.nunique() <= 1:
        print("Series is constant — cannot compute ACF/PACF.")
        return
    
    # ======= II. Compute PACF values =======
    pacf_vals = tm.get_partial_autocorrelation(series, lags)
    N = len(series)
    confidence_interval = 1.96 / np.sqrt(N)

    # ======= III. Plot PACF =======
    plt.figure(figsize=figsize)
    plt.stem(range(1, len(pacf_vals)+1), pacf_vals, basefmt=" ")
    plt.axhline(confidence_interval, linestyle='--', color='red', alpha=0.5)
    plt.axhline(-confidence_interval, linestyle='--', color='red', alpha=0.5)
    plt.axhline(0, linestyle='--', color='gray')
    plt.title(title)
    plt.ylabel("PACF")
    plt.xlabel("Lag")
    plt.tight_layout()
    plt.show()

#*____________________________________________________________________________________ #
def plot_volatility(
    series: pd.Series, 
    rolling_window: int = 20, 
    quantile: float = 0.75,
    figsize: tuple = (17, 7),
    title: str = 'Volatility Clustering Visualization',
):
    """
    Plot the rolling volatility of a time series and highlight high volatility regimes.

    Parameters:
        - series (pd.Series): The time series data to analyze.
        - rolling_window (int): The window size for rolling volatility calculation.
        - quantile (float): The quantile threshold for high volatility detection.
        - figsize (tuple): The size of the figure.
        - title (str): The title of the plot.
    
    Returns:
        - None: Displays the plot.
    """
    # ======= 0. Helper Function =======
    def shade_intervals(ax, mask, color='red', alpha=0.15):
        mask = mask.astype(int)
        changes = mask.diff().fillna(0).abs()
        starts = mask[(changes == 1) & (mask == 1)].index
        ends = mask[(changes == 1) & (mask == 0)].index
        
        # If mask starts True from beginning
        if mask.iloc[0] == 1:
            starts = starts.insert(0, mask.index[0])
        # If mask ends True at the end
        if mask.iloc[-1] == 1:
            ends = ends.append(pd.Index([mask.index[-1]]))
        
        for start, end in zip(starts, ends):
            for a in ax:
                a.axvspan(start, end, color=color, alpha=alpha)

    # ======= I. Compute Rolling Volatility =======
    rolling_vol = series.rolling(window=rolling_window).std()
    threshold = rolling_vol.quantile(quantile)
    
    # ======= II. Plot Volatility =======
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # 1. Plot original series
    ax[0].plot(series, color='blue', label='Original Series')
    ax[0].set_ylabel('Series Value')
    ax[0].legend()
    ax[0].grid(True)
    
    # 2. Plot rolling volatility
    ax[1].plot(rolling_vol, color='green', label=f'Rolling Volatility (std, window={rolling_window})')
    ax[1].set_ylabel('Volatility')
    ax[1].legend()
    ax[1].grid(True)
    
    # ======= III. Highlight High Volatility Regimes =======
    high_vol_mask = rolling_vol > threshold
    shade_intervals(ax, high_vol_mask)
    
    plt.xlabel('Date')
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



#! ==================================================================================== #
#! ============================== Labels Visualization  =============================== #
def plot_series_labels(
    series: pd.Series,
    label_series: pd.Series,
    series_name: str = "close",
    label_name: str = "label",
    marker_size: int = 50,
    colormap_continuous=cm.viridis,
):
    """
    Plot a time series with overlaid label markers. Supports both categorical and continuous labels.

    Parameters:
        series (pd.Series): Time series to plot (indexed by datetime).
        label_series (pd.Series): Labels aligned with the series (categorical or continuous).
        series_name (str): Title and Y-axis label.
        label_name (str): Label used in the legend and colorbar.
        marker_size (int): Size of label scatter markers.
        colormap_continuous: Matplotlib colormap for continuous values (default: viridis).
    """
    def generate_color_dict(unique_labels):
        """
        Maps labels to perceptually consistent diverging colors:
        - Negative → red
        - 0 → blue (neutral)
        - Positive → green
        """
        unique_labels = sorted(unique_labels)
        color_dict = {}

        if len(unique_labels) == 1:
            return {unique_labels[0]: '#000000'}

        min_label, max_label = min(unique_labels), max(unique_labels)
        norm_labels = [
            (label - min_label) / (max_label - min_label) if min_label != max_label else 0.5
            for label in unique_labels
        ]

        diverging_cmap = mcolors.LinearSegmentedColormap.from_list(
            "custom_diverging",
            ["red", "blue", "green"],
            N=256
        )

        for label, norm_val in zip(unique_labels, norm_labels):
            color = mcolors.to_hex(diverging_cmap(norm_val))
            color_dict[label] = color

        return color_dict

    # ================= Plotting =================
    fig, ax = plt.subplots(figsize=(17, 5))
    ax.plot(series.index, series, label=series_name, color="black", linewidth=2, zorder=10)

    label_series = label_series.reindex(series.index)  # align just in case
    unique_labels = np.sort(label_series.dropna().unique())

    if len(unique_labels) <= 10:
        # ===== Categorical Labels =====
        color_dict = generate_color_dict(unique_labels)
        for label in unique_labels:
            mask = label_series == label
            ax.scatter(
                series.index[mask],
                series[mask],
                color=color_dict[label],
                label=f"{label_name}: {label}",
                s=marker_size,
                alpha=0.8
            )
    else:
        # ===== Continuous Labels =====
        norm = plt.Normalize(label_series.min(), label_series.max())
        sc = ax.scatter(
            series.index,
            series,
            c=label_series,
            cmap=colormap_continuous,
            norm=norm,
            s=marker_size,
            alpha=0.8,
            label=label_name
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(label_name)

    ax.set_title(f"{series_name} with {label_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel(series_name)
    ax.grid(True)
    ax.legend(loc='upper left', fontsize='small', frameon=True)
    plt.tight_layout()
    plt.show()

#*____________________________________________________________________________________ #
def plot_feature_vs_label(
    feature_series: pd.Series, 
    label_series: pd.Series, 
    feature_name: str = None
) -> None:
    """
    Visualize and analyze the relationship between a numerical feature and its corresponding labels.

    This function produces three visualizations:
    1. Time-series plot of the feature with overlaid label markers.
    2. Boxplot of the feature values grouped by label.
    3. Correlation heatmap between the feature and label.

    Parameters:
        feature_series (pd.Series): Numerical feature Series indexed by time.
        label_series (pd.Series): Label Series indexed identically to `feature_series`, with values in {-1, 0, 1}.
        feature_name (str, optional): Name of the feature for titles/labels. If None, a generic name is used.

    Returns:
        None
    """
    if feature_name is None:
        feature_name = "Feature"

    # ======= I. Time-series visualization with label markers =======
    plt.figure(figsize=(17, 5))
    plt.plot(feature_series.index, feature_series, label=feature_name, linewidth=2)

    # Unique label legend tracker to avoid duplicates
    shown_labels = set()
    for i, label in label_series.items():
        color = {1: 'green', 0: 'black', -1: 'red'}.get(label, 'gray')
        label_str = {
            1: 'Reg: Upward Movement',
            0: 'Reg: Neutral Movement',
            -1: 'Reg: Downward Movement'
        }.get(label, f'Label: {label}')
        if label_str not in shown_labels:
            plt.scatter(i, 0, color=color, label=label_str, s=10, zorder=5)
            shown_labels.add(label_str)
        else:
            plt.scatter(i, 0, color=color, s=10, zorder=5)

    plt.title(f'{feature_name} over Time with Labels')
    plt.xlabel('Date')
    plt.ylabel(f'{feature_name} Value')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # ======= II. Boxplot by label =======
    label_feature_df = pd.DataFrame({'label': label_series, feature_name: feature_series})
    plt.figure(figsize=(17, 5))
    sns.boxplot(x='label', y=feature_name, data=label_feature_df)
    plt.title(f'Boxplot of {feature_name} by Label')
    plt.xlabel('Labels')
    plt.ylabel(f'{feature_name} Values')
    plt.tight_layout()
    plt.show()

    # ======= III. Correlation heatmap =======
    light_green = (0, 0.7, 0.3, 0.2)
    colors = [(0, 'green'), (0.5, light_green), (0.99, 'green'), (1, 'grey')]
    cmap_name = 'green_white'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=1000)

    corr_matrix = label_feature_df.corr()
    plt.figure(figsize=(17, 3))
    sns.heatmap(corr_matrix, annot=True, cmap=cm, vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
    plt.title(f'Correlation between {feature_name} and Label')
    plt.tight_layout()
    plt.show()



