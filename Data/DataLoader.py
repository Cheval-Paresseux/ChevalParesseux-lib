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


def load_dataList(ticker_list: list = None):
    # Define the paths to the data directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_nyse_dir = os.path.join(current_dir, 'dataNYSE')
    data_nasdaq_dir = os.path.join(current_dir, 'dataNASDAQ')

    # If ticker_list is None, load all available data
    if ticker_list is None:
        ticker_list = []

        # List all CSV files in the NYSE directory
        for file_name in os.listdir(data_nyse_dir):
            if file_name.endswith('.csv'):
                ticker_list.append(file_name.replace('.csv', ''))

        # List all CSV files in the NASDAQ directory
        for file_name in os.listdir(data_nasdaq_dir):
            if file_name.endswith('.csv'):
                ticker_list.append(file_name.replace('.csv', ''))

        # Remove duplicates by converting the list to a set and back to a list
        ticker_list = list(set(ticker_list))

    data_list = {}
    for ticker in ticker_list:
        data_list[ticker] = (load_data(ticker))

    return data_list
