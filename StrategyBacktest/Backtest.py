import sys
sys.path.append("../")
from StrategyBacktest import Strategies as st



class Backtest():
    def __init__(self, strategy: st.Strategy):
        self.strategy = strategy
        self.data = None
        self.processed_data = None

        self.signals_dfs = None
        self.operations_df = None

        self.ticker = None
        self.start_date = None
        self.end_date = None
        self.strategy_params = None
    
    #*____________________________________________________________________________________ #
    def set_backtestParams(self, ticker: str, start_date: str, end_date: str, strategy_params: dict):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.strategy_params = strategy_params
    
    #*____________________________________________________________________________________ #
    def run_strategy(self):
        self.strategy.set_params(self.strategy_params)
        data = self.strategy.load_data(self.ticker, self.start_date, self.end_date)
        processed_data = self.strategy.process_data(data)
        signals_dfs, operations_df = self.strategy.operate(processed_data)

