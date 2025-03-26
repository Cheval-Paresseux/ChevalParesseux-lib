import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

#! ==================================================================================== #
#! ================================= Base Model ======================================= #
class Labeller(ABC):
    @abstractmethod
    def __init__(
        self, 
        data: pd.Series, 
        params: dict, 
        n_jobs: int = 1
    ):
        # ======= I. Initialize Class =======
        self.data = data
        self.params = params
        self.n_jobs = n_jobs

        # ======= II. Initialize Auxilaries =======
        self.processed_data = None
        self.labels = None
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def process_data(self):
        pass
    
    #?____________________________________________________________________________________ #
    @abstractmethod
    def get_labels(self):
        pass
    
    #?____________________________________________________________________________________ #
    def fit(self):
        pass
    
    #?____________________________________________________________________________________ #
    def extract(self):
        if self.labels:
            return self.labels
        else:
            params_grid = extract_universe(self.params)
            labels = Parallel(n_jobs=self.n_jobs)(delayed(self.get_labels)(**params) for params in params_grid)
            labels_df = pd.concat([series.to_frame().rename(lambda col: f"set_{i}", axis=1) for i, series in enumerate(labels)], axis=1)
            
            self.labels = labels_df

            return labels_df
    
    #?____________________________________________________________________________________ #
    def plot_labels(self, labels_series: pd.Series):
        # ======= I. Extract the series =======
        price_series = self.data
        label_series = labels_series.reindex(price_series.index, fill_value=np.nan)

        
        # ======= II. Plot the labels =======
        plt.figure(figsize=(17, 4))
        plt.plot(price_series, label="Prix", color="blue", linewidth=1)
        
        plt.scatter(price_series.index[label_series == 1], 
                    price_series[label_series == 1], 
                    color='green', label="Tendance haussière (+1)", marker="^", s=50)
        
        plt.scatter(price_series.index[label_series == -1], 
                    price_series[label_series == -1], 
                    color='red', label="Tendance baissière (-1)", marker="v", s=50)
        
        plt.scatter(price_series.index[label_series == 0], 
                    price_series[label_series == 0], 
                    color='gray', label="Neutre (0)", marker="o", s=10, alpha=0.5)

        plt.title("Prix avec Labels de Tendance")
        plt.xlabel("Date")
        plt.ylabel("Prix")
        plt.legend()
        plt.grid(True)
        plt.show()



#! ==================================================================================== #
#! ================================= Helper Functions ================================= #
def extract_universe(params_grid: dict): 
    # ======= 0. Define recursive function to generate all combinations =======
    def recursive_combine(keys, values, index, current_combination, params_list):
        if index == len(keys):
            # Base case: all parameters have been assigned a value
            params_list.append(current_combination.copy())
            return

        key = keys[index]
        for value in values[index]:
            current_combination[key] = value
            recursive_combine(keys, values, index + 1, current_combination, params_list)

    # ======= I. Initialize variables =======
    keys = list(params_grid.keys())
    values = list(params_grid.values())
    params_list = []

    # ======= II. Generate all combinations =======
    recursive_combine(keys, values, 0, {}, params_list)

    return params_list

#*____________________________________________________________________________________ #
def trend_filter(label_series: pd.Series, window: int):
    # ======= I. Create an auxiliary DataFrame =======
    auxiliary_df = pd.DataFrame()
    auxiliary_df["label"] = label_series
    
    # ======= II. Create a group for each label and extract size =======
    auxiliary_df["group"] = (auxiliary_df["label"] != auxiliary_df["label"].shift()).cumsum()
    group_sizes = auxiliary_df.groupby("group")["label"].transform("size")

    # ======= III. Filter the labels based on the group size =======
    auxiliary_df["label"] = auxiliary_df.apply(lambda row: row["label"] if group_sizes[row.name] >= window else 0, axis=1)
    labels_series = auxiliary_df["label"]
    
    return labels_series
