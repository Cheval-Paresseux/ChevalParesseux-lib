import os
import pandas as pd


def load_data(ticker: str):
    # ======= I. Define the paths to the data directories =======
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_nyse_dir = os.path.join(current_dir, 'dataNYSE')
    data_nasdaq_dir = os.path.join(current_dir, 'dataNASDAQ')

    # ======= II. Construct the full paths to the CSV files  =======
    file_name = f'{ticker}.csv'

    csv_path_nyse = os.path.join(data_nyse_dir, file_name)
    csv_path_nasdaq = os.path.join(data_nasdaq_dir, file_name)

    # ======= III. Load the data =======
    try:
        data = pd.read_csv(csv_path_nyse, index_col=0, parse_dates=True)
    except FileNotFoundError:
        data = pd.read_csv(csv_path_nasdaq, index_col=0, parse_dates=True)

    return data