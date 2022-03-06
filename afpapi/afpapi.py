import numpy as np
import os
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "rets.csv")
FACTOR_DATA_PATH = os.path.join(this_dir, "factors.csv")
WEIGHTS_DATA_PATH = this_dir
COV_MATRIX_DATA_PATH = this_dir
SPY_DATA_PATH = this_dir
OUTPUT_DIR = this_dir

class AFP:
    def __init__(self, data_path=None, start_date='2001-01-03', end_date='2021-12-31',
                 initial_price=100, rolling=5*252, t_costs=0.0010, load_weights=True):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_price = initial_price
        self.rolling = rolling
        self.t_costs = t_costs
        self.initialize_data(data_path)
        self.initialize_prices()
        self.calculate_equal_weight_returns()

    def initialize_data(self, data_path=None):
        if data_path is None:
            data = pd.read_csv(DATA_PATH, index_col=0)
            data.index = pd.to_datetime(data.index)
            factor_data = pd.read_csv(FACTOR_DATA_PATH, index_col=0)
            factor_data.index = pd.to_datetime(factor_data.index)
        else:
            data = pd.read_csv(f'{data_path}/rets.csv', index_col=0)
            data.index = pd.to_datetime(data.index)
            factor_data = pd.read_csv(f'{data_path}/factors.csv', index_col=0)
            factor_data.index = pd.to_datetime(factor_data.index)
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
        self.factors = factor_data

    def initialize_prices(self):
        rets = self.rets
        prices = self.initial_price * (1 + rets).cumprod()
        self.prices = prices

    def calculate_mean_variance_returns(self, risk_aversion=1):
        start_index = self.rets.index.get_loc(self.start_date)
        if isinstance(start_index, np.ndarray):
            start_index = start_index[0]
        end_index = self.rets.index.get_loc(self.end_date)
        if isinstance(end_index, np.ndarray):
            end_index = end_index[0]

        rolling = self.rolling
        rets = self.rets
        returns = []
        weights = self.mv_weights
        for ix in range(start_index, end_index+1):
            d = rets.index[ix]
            w = weights.loc[d]
            daily_rets = rets.loc[d]
            if ix == end_index:
                t_costs = 0
            else:
                d_next = rets.index[ix+1]
                w_next = weights.loc[d_next]
                sum_end_of_day_weights = w.values.dot(1 + daily_rets)
                end_of_day_weights = np.multiply(w.values, 1 + daily_rets)/sum_end_of_day_weights
                turnover = np.abs(w_next - end_of_day_weights).sum()
                t_costs = self.t_costs * turnover
            ret = daily_rets.dot(w.values) #- t_costs
            returns.append({'date': d, 'mv_returns': ret, 'turnover': turnover})

        mv_rets_df = pd.DataFrame(returns)
        mv_rets_df.set_index('date', inplace=True)
        self.mv_rets_df = mv_rets_df

    def calculate_equal_weight_returns(self):
        start_index = self.rets.index.get_loc(self.start_date)
        if isinstance(start_index, np.ndarray):
            start_index = start_index[0]
        end_index = self.rets.index.get_loc(self.end_date)
        if isinstance(end_index, np.ndarray):
            end_index = end_index[0]
        rets = self.rets
        returns = []
        n_assets = len(rets.columns)
        weights = pd.Series([1/n_assets] * n_assets)
        for ix in range(start_index, end_index+1):
            daily_rets = rets.loc[rets.index[ix]]
            sum_end_of_day_weights = weights.values.dot(1 + daily_rets)
            end_of_day_weights = np.multiply(weights.values, 1 + daily_rets)/sum_end_of_day_weights
            turnover = np.abs(weights.values - end_of_day_weights).sum()
            t_costs = self.t_costs * turnover
            ret = daily_rets.mean() - t_costs
            returns.append({'date': rets.index[ix], 'eq_returns': ret, 'turnover': turnover})

        eq_rets_df = pd.DataFrame(returns)
        eq_rets_df.set_index('date', inplace=True)
        self.eq_rets_df = eq_rets_df

    def calculate_spy_returns(self, loc=None):
        if loc is None:
            loc = f'{SPY_DATA_PATH}/spy.csv'
        else:
            loc = f'{loc}/spy.csv'
        spy = pd.read_csv(loc)
        spy['Date'] = pd.to_datetime(spy['Date'])
        spy = spy.sort_values('Date')
        spy.set_index('Date', inplace=True)
        spy['turnover'] = np.nan
        self.sp_rets_df = spy

    def compare(self, weights=None):
        rets = self.rets
        start_index = rets.index.get_loc(self.start_date)[0]
        end_index = rets.index.get_loc(self.end_date)[0]
        returns = []
        merged_df = None
        if hasattr(self, 'eq_rets_df'):
            merged_df = self.eq_rets_df['eq_returns']
        if hasattr(self, 'mv_rets_df'):
            merged_df = pd.merge(merged_df, self.mv_rets_df['mv_returns'], how='inner', left_index=True, right_index=True)
        if hasattr(self, 'sp_rets_df'):
            merged_df = pd.merge(merged_df, self.sp_rets_df['sp_returns'], how='inner', left_index=True, right_index=True)

        if weights is not None:
            for ix in range(start_index, end_index+1):
                d = rets.index[ix]
                w = weights.loc[d]
                daily_rets = rets.loc[d]
                if ix == end_index:
                    t_costs = 0
                else:
                    d_next = rets.index[ix+1]
                    w_next = weights.loc[d_next]
                    sum_end_of_day_weights = w.values.dot(1 + daily_rets)
                    end_of_day_weights = np.multiply(w.values, 1 + daily_rets)/sum_end_of_day_weights
                    turnover = np.abs(w_next - end_of_day_weights).sum()
                    t_costs = self.t_costs * turnover
                ret = daily_rets.dot(w.values) - t_costs
                returns.append({'date': d, 'drl_returns': ret, 'turnover': turnover})

            drl_rets_df = pd.DataFrame(returns)
            drl_rets_df.set_index('date', inplace=True)
            self.drl_rets_df = drl_rets_df
            merged_df = pd.merge(merged_df, drl_rets_df['drl_returns'], how='inner', left_index=True, right_index=True)

        self.merged_df = merged_df
        self.cum_rets_df = ((1+self.merged_df).cumprod()-1)

    def generate_data(self, output_dir=None):
        if output_dir is None:
            output_dir = OUTPUT_DIR
        rets = self.rets
        start_date = self.start_date
        end_date = self.end_date
        start_index = rets.index.get_loc(start_date)[0]
        end_index = rets.index.get_loc(end_date)[0]
        rolling = self.rolling
        weights_list = []
        n = len(rets.columns)
        w_prev = np.array([1/n]*n)
        weight_df = pd.DataFrame(columns=rets.columns)
        for ix in tqdm(range(start_index, end_index+1)):
            if rolling != 'max':
                start_ix = ix - rolling - 1 # subtracting 1 to get "rolling" data points not including today
            else:
                start_ix = 0
            data = rets.loc[rets.index[start_ix:ix-1]]
            mu = mean_historical_return(data, returns_data=True).values
            sigma = risk_matrix(data, returns_data=True)
            w = cp.Variable(n)
            ret = mu.T@w
            risk = cp.quad_form(w, sigma)/2
            prob = cp.Problem(cp.Maximize(ret - risk_aversion*risk - 0.0005 * cp.norm(w - w_prev, 1)),
                             [cp.sum(w) == 1, w >= 0])
            prob.solve(solver=cp.SCS)
            weight_dict = {col: weight for col,weight in zip(rets.columns, w.value)}
            weight_dict['Date'] = rets.index[ix]
            weights_list.append(weight_dict)
            w_prev = w.value
            sigma_df = pd.DataFrame(sigma, index=rets.columns, columns=rets.columns)
            sigma_df.to_csv(f'{output_dir}/{rets.index[ix]}_cov_matrix.csv')

        weight_df = pd.DataFrame(weights_list)
        weight_df = weight_df.sort_values('Date')
        weight_df.set_index('Date', inplace=True)
        weight_df.to_csv(f'{output_dir}/mv_weights.csv')

    def get_risk_matrix(self, date=None, loc=None):
        if loc is None:
            loc = COV_MATRIX_DATA_PATH
        if date is None:
            print('You must provide a date.')
            return
        rm = pd.read_csv(f'{loc}/{date}_cov_matrix.csv', index_col=0)
        return rm

    def load_weights(self, loc=None):
        if loc is None:
            loc = WEIGHTS_DATA_PATH
        w = pd.read_csv(f'{loc}/mv_weights.csv', index_col=0)
        w.index = pd.to_datetime(w.index)
        w.sort_index(inplace=True)
        self.mv_weights = w

    def create_tear_sheet(self):
        cols = []
        if hasattr(self, 'eq_rets_df'):
            cols.append('eq')
        if hasattr(self, 'mv_rets_df'):
            cols.append('mv')
        if hasattr(self, 'drl_rets_df'):
            cols.append('drl')
        if hasattr(self, 'sp_rets_df'):
            cols.append('sp')
        data = []
        for col in cols:

            asset_df = self.__dict__[f'{col}_rets_df']
            series = asset_df[f'{col}_returns']
            avg_turnover = asset_df['turnover'].mean()
            cum_ret = (1+series).cumprod() - 1
            sub_df = pd.DataFrame(index=series.index)
            sub_df['returns'] = series
            sub_df = pd.merge(self.factors, sub_df, how='inner', left_index=True, right_index=True)
            sub_df['excess_returns'] = sub_df['returns'] - sub_df['RF']
            mu_excess = mean_historical_return(sub_df['excess_returns'], returns_data=True)[0]
            excess_std = sub_df['excess_returns'].std()*np.sqrt(252)
            sharpe = mu_excess / excess_std

            mu = mean_historical_return(series, returns_data=True)[0]
            std = series.std()*np.sqrt(252)
            d = {
                'strategy': col,
                'mean_return': mu,
                'std_dev': std,
                'max_drawdown': ((1 + cum_ret)/(1 + cum_ret.cummax()) - 1).min(),
                'sharpe_ratio': sharpe,
                'daily_turnover': avg_turnover
            }
            data.append(d)

        r_df = pd.DataFrame(data)
        r_df.set_index('strategy', inplace=True)
        self.tear_sheet = r_df
