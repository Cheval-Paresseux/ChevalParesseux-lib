import os
import pandas as pd


def load_data(
    ticker: str
) -> pd.DataFrame:
    """
    Load data from CSV files for a given ticker symbol.
    
    Parameters:
        - ticker (str): The ticker symbol of the stock.
    
    Returns:
        - pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    # ======= I. Define the paths to the data directories =======
    current_dir = os.path.dirname(os.path.abspath(__file__))

    data_nyse_dir = os.path.join(current_dir, 'NYSE')
    data_nasdaq_dir = os.path.join(current_dir, 'NASDAQ')
    data_etfs_dir = os.path.join(current_dir, 'ETFs')

    # ======= II. Construct the full paths to the CSV files  =======
    file_name = f'{ticker}.csv'

    csv_path_nyse = os.path.join(data_nyse_dir, file_name)
    csv_path_nasdaq = os.path.join(data_nasdaq_dir, file_name)
    csv_path_etfs = os.path.join(data_etfs_dir, file_name)

    # ======= III. Load the data =======
    data = None
    for path in [csv_path_nyse, csv_path_nasdaq, csv_path_etfs]:
        try:
            data = pd.read_csv(path)
            break  
        except FileNotFoundError:
            continue
    
    if data is None:
        raise FileNotFoundError(f"Data for ticker '{ticker}' not found in any directory.")
    
    return data


def load_dataList(
    ticker_list: list = None
) -> dict:
    """
    Load data for a list of ticker symbols. If no list is provided, load all available data.

    Parameters:
        - ticker_list (list): A list of ticker symbols. If None, load all available data.
    
    Returns:
        - dict: A dictionary where keys are ticker symbols and values are DataFrames.
    """
    # ======= I. Define the paths to the data directories =======
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_nyse_dir = os.path.join(current_dir, 'NYSE')
    data_nasdaq_dir = os.path.join(current_dir, 'NASDAQ')
    data_etfs_dir = os.path.join(current_dir, 'ETFs')

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
            
        # List all CSV files in the ETfs directory
        for file_name in os.listdir(data_etfs_dir):
            if file_name.endswith('.csv'):
                ticker_list.append(file_name.replace('.csv', ''))

        # Remove duplicates by converting the list to a set and back to a list
        ticker_list = list(set(ticker_list))

    data_list = {}
    for ticker in ticker_list:
        data_list[ticker] = (load_data(ticker))

    return data_list
