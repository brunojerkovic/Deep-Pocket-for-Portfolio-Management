import os
import numpy as np
from datetime import datetime
from empyrical import sharpe_ratio


class Metrics:
    def __init__(self, period, dataset, p_0, save_folder):
        self.period_start = datetime.strptime(period[0], '%Y-%m-%d')
        self.period_end = datetime.strptime(period[1], '%Y-%m-%d')
        self.investment_duration = self.period_end - self.period_start
        self.dataset = dataset
        self.p_0 = p_0
        self.save_folder = save_folder
        self.save_file = os.path.join(self.save_folder, 'experiment.json')

    def return_on_investment(self, portfolio_values: list) -> dict:
        '''
        Calculates the return on investment
        :param portfolio_values: List of portfolio values.
        :return: Return on investment.
        '''
        p_f = portfolio_values[-1]
        roi = (p_f - self.p_0) / self.p_0
        return {'roi': roi}

    @staticmethod
    def sharpe_ratio(portfolio_values: list) -> dict:
        '''
        Calculates the daily and annual Sharpe ratio.
        :param portfolio_values: list of values of the portfolio
        :return: Annual and daily Sharpe ratios
        '''
        port_val_np = np.array(portfolio_values)
        port_change = np.diff(port_val_np) / port_val_np[:-1]
        daily_sharpe = port_change.mean() / port_change.std(ddof=1)
        annual_sharpe = 252 ** (1 / 2) * daily_sharpe

        daily_sharpe = daily_sharpe if str(daily_sharpe) != str(np.nan) else 0.
        annual_sharpe = annual_sharpe if str(annual_sharpe) != str(np.nan) else 0.
        return {'daily_sharpe': daily_sharpe, 'annual_sharpe': annual_sharpe}

    @staticmethod
    def sharpe_ratio2(portfolio_values: list) -> dict:
        port_val_np = np.array(portfolio_values)
        sharpe = sharpe_ratio(np.log(port_val_np))
        return {'daily_sharpe': sharpe}

    def max_drawdown(self, idx: int) -> dict:
        '''
        Calculates the maximum drawdown of the closing prices
        :param idx: idx of the current datapoint
        :return: maximum-drawdown value
        '''
        closing_prices = self.dataset.closing_prices[idx]
        min_price = np.min(closing_prices)
        max_price = np.max(closing_prices)

        mdd = (max_price - min_price) / max_price
        return {'mdd': np.float64(mdd)}
