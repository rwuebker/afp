import os
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "rets.csv")
VALUE_DATA_PATH = os.path.join(this_dir, "value_rets.csv")

class AFP:
    def __init__(self, data_path=None, value_data_path=None, start_date='2000-01-03', end_date='2002-12-31',
                 initial_price=100, rolling='max', t_costs=0.0005):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_price = initial_price
        self.rolling = rolling
        self.t_costs = t_costs
        self.initialize_data(data_path)
        self.initialize_value_data(value_data_path)
        self.initialize_prices()
        self.calculate_equal_weight_returns()
        self.calculate_value_weighted_returns()

    def initialize_data(self, data_path=None):
        if data_path is None:
            data = pd.read_csv(DATA_PATH, index_col=0)
            data.index = pd.to_datetime(data.index)
        else:
            data = pd.read_csv(data_path, index_col=0)
            data.index = pd.to_datetime(data.index)

        try:
            data.index.get_loc(self.start_date)
        except KeyError as e:
            print(f'ERROR: start_date of {self.start_date} is not in the data')
            raise e

        try:
            data.index.get_loc(self.end_date)
        except KeyError as e:
            print(f'ERROR: end_date of {self.end_date} is not in the data.')
            raise e

        self.rets = data

    def initialize_value_data(self, value_data_path=None):
        if value_data_path is None:
            data = pd.read_csv(VALUE_DATA_PATH, index_col=0)
            data.index = pd.to_datetime(data.index)
        else:
            data = pd.read_csv(value_data_path, index_col=0)
            data.index = pd.to_datetime(data.index)

        try:
            data.index.get_loc(self.start_date)
        except KeyError as e:
            print(f'ERROR: start_date of {self.start_date} is not in the data')
            raise e

        try:
            data.index.get_loc(self.end_date)
        except KeyError as e:
            print(f'ERROR: end_date of {self.end_date} is not in the data.')
            raise e

        self.value_rets = data

    def initialize_prices(self):
        rets = self.rets
        prices = self.initial_price * (1 + rets).cumprod()
        self.prices = prices

    def calculate_mean_variance_returns(self, method='max_quadratic_utility', risk_aversion=20,
                                        market_neutral=True):
        start_index = self.rets.index.get_loc(self.start_date)
        end_index = self.rets.index.get_loc(self.end_date)
        rolling = self.rolling
        rets = self.rets
        first_date = rets.index[0]
        returns = []
        for ix in range(start_index, end_index+1):
            if rolling != 'max':
                start_ix = ix - rolling - 1 # subtracting 1 to get "rolling" data points not including today
            else:
                start_ix = 0
            data = rets.loc[rets.index[start_ix:ix-1]]
            mu = mean_historical_return(data, returns_data=True)
            sigma = CovarianceShrinkage(data, returns_data=True).ledoit_wolf()
            ef = EfficientFrontier(mu, sigma)
            if method == 'max_quadratic_utility':
                weights = ef.max_quadratic_utility(risk_aversion=risk_aversion, market_neutral=market_neutral)
            elif method == 'max_sharpe':
                weights = ef.max_sharpe()

            weights_df = pd.Series(weights)
            ret = rets.loc[rets.index[ix]] - self.t_costs
            ret = ret.dot(weights_df)
            returns.append({'date': rets.index[ix], 'mvo_return': ret})

        mv_rets_df = pd.DataFrame(returns)
        mv_rets_df.set_index('date', inplace=True)
        self.mv_rets_df = mv_rets_df

    def calculate_equal_weight_returns(self):
        start_index = self.rets.index.get_loc(self.start_date)
        end_index = self.rets.index.get_loc(self.end_date)
        rets = self.rets
        returns = []
        n_assets = len(rets.columns)
        weights = pd.Series([1/n_assets] * n_assets)
        for ix in range(start_index, end_index+1):
            ret = rets.loc[rets.index[ix]].mean() - self.t_costs
            returns.append({'date': rets.index[ix], 'ew_return': ret})

        eq_rets_df = pd.DataFrame(returns)
        eq_rets_df.set_index('date', inplace=True)
        self.eq_rets_df = eq_rets_df

    def calculate_value_weighted_returns(self):
        start_index = self.rets.index.get_loc(self.start_date)
        end_index = self.rets.index.get_loc(self.end_date)
        rets = self.value_rets
        returns = []
        n_assets = len(rets.columns)
        for ix in range(start_index, end_index+1):
            ret = (rets.loc[rets.index[ix]]).mean() - self.t_costs
            returns.append({'date': rets.index[ix], 'vw_return': ret})

        vw_rets_df = pd.DataFrame(returns)
        vw_rets_df.set_index('date', inplace=True)
        self.vw_rets_df = vw_rets_df


    def compare(self, weights):
        if not hasattr(self, 'mv_rets_df'):
            raise Exception('You never calculated your mean-variance weights.')

        rets = self.rets
        start_index = self.rets.index.get_loc(self.start_date)
        end_index = self.rets.index.get_loc(self.end_date)
        returns = []
        for ix in range(start_index, end_index+1):
            d = rets.index[ix]
            w = weights.loc[d]
            ret = w.dot(rets.loc[d] - self.t_costs)
            returns.append({'date': d, 'drl_return': ret})


        merged_df = pd.merge(self.mv_rets_df, self.eq_rets_df, left_index=True, right_index=True)
        drl_rets_df = pd.DataFrame(returns)
        drl_rets_df.set_index('date', inplace=True)
        merged_df = pd.merge(merged_df, drl_rets_df, left_index=True, right_index=True)
        if hasattr(self, 'vw_rets_df'):
            merged_df = pd.merge(merged_df, self.vw_rets_df, left_index=True, right_index=True)
        self.merged_rets_df = merged_df
        self.cum_rets_df = ((1+self.merged_rets_df).cumprod()-1)
