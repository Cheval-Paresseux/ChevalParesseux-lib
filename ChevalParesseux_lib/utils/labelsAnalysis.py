import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#! ==================================================================================== #
#! ============================== Labels Visualization  =============================== #
def plot_price_with_labels(price_series: pd.Series, label_series: pd.Series):

    plt.figure(figsize=(17, 5))
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

    plt.title(f"Price with Labels {label_series.name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
    
#*____________________________________________________________________________________ #