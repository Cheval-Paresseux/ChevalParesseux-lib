# ChevalParesseux-lib

**ChevalParesseux-lib** is a Python library designed to industrialize the research process for systematic and algorithmic trading strategies. The core philosophy behind this library is that any trading strategy follows a standardized workflow: gathering information at time \( t \), processing this information, extracting signals, applying a strategy layer, and executing trades.

The library is divided into two main components: "Betting" and "Hedging." The "Betting" component focuses on generating alpha by predicting the direction of assets, while the "Hedging" component provides tools for risk management and portfolio optimization.

## Philosophy

The primary goal of industrializing quantitative research in finance is to create specialized sectors, each focused on a specific task. This specialization helps avoid emotional involvement with results and ensures that each sector can operate independently and efficiently. To achieve this, **ChevalParesseux-lib** provides a well-structured framework with standardized solutions, allowing each sector to output its results using a unified wrapper. This approach minimizes friction and enhances collaboration between different parts of the trading process.

## Components

### I. Betting

The "Betting" component encompasses all sections related to finding alpha in financial markets.

#### 1. Data Processing

- **Features**: This section focuses on extracting meaningful features from available data. Since financial markets deal with time series data, the features should also be time series (even if they are constant over certain periods).

  The output of this section should be a class that inherits from the `Feature(ABC)` class, following this workflow:

  ```python
  feature_object = feature_xx(name='feature_name', n_jobs=n_jobs)
  feature_object.set_params(**feature_params)
  features_df = feature_object.extract(data=time_series)
  # features_df being a DataFrame containing series for each set of parameters.

- **Labelling**: This section focuses on extracting meaningful labels from available data. It aims to train Machine Learning models and should also be time series (even if they are constant over certain periods, ex: regimes).

    The output of this section should be a class that inherits from the `Labeller(ABC)` class, following this workflow:

  ```python
  labeller_object = feature_xx(name='labeller_name', n_jobs=n_jobs)
  labeller_object.set_params(**labels_params)
  labels_df = labeller_object.extract(data=time_series)
  # labels_df being a DataFrame containing series for each set of parameters.

