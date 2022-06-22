import os
import pandas as pd
from datetime import datetime, timedelta
from functools import reduce
import numpy as np


class PreprocessedDataset(dict):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.n_stocks = 0
        self.load_datasets()  # Keys are set here

        # Copy of some attributes (making a copy uses RAM, but makes access easier)
        # self.dates = self.get_all_dates()
        # self.closing_prices = self.get_closing_prices()

    def load_datasets(self):
        # Load ae
        self.n_stocks = len(os.listdir(self.folder_path))
        for idx, filepath in enumerate(os.listdir(self.folder_path)):
            df = pd.read_csv(os.path.join(self.folder_path, filepath))
            unnamed_col = [col for col in df.columns if col.lower().startswith('unnamed')]
            df.drop(columns=unnamed_col, inplace=True)
            self[filepath.removesuffix('.csv')] = df

        # Parse stock's date into uniform formatting
        for short_name, dataset in self.items():
            date_format = '%Y-%m-%d %H:%M:%S' if '00:00:00' in dataset['date'].iloc[0] else '%Y-%m-%d'
            datetime_series = dataset['date'].apply(datetime.strptime, args=(date_format,))
            datestr_series = datetime_series.apply(datetime.strftime, args=('%Y-%m-%d',))
            self[short_name]['date'] = datestr_series

        # Only keep samples whose dates are present in ALL stocks
        # Reduce applies the merge function on 1st and 2nd df-s, then on its result and 3rd df, and so on
        #dates = reduce(lambda df1, df2: pd.merge(df1, df2, on='date'), list(self.values()))['date'].tolist()
        dates = reduce(lambda df1, df2: pd.merge(df1['date'], df2['date']), list(self.values())).values.squeeze().tolist()
        for short_name, dataset in self.items():
            self[short_name] = dataset[dataset['date'].isin(dates)]

    def get_shortnames(self):
        return list(self.keys())

    def get_close(self, stock_shortname):
        """Get all closing prices (and dates) for a particular stock"""
        return self[stock_shortname][['date', 'close']]

    def date_to_idx(self, date):
        while not (date in self.dates):
            date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        idx = self.dates.index(date)
        return idx

    @property
    def dates(self) -> list:
        """Returns a list of the dates for the current market"""
        dates = list(self.values())[0]['date'].tolist()
        return dates

    @property
    def closing_prices(self) -> np.ndarray:
        """Returns numpy list of closing prices for each date"""
        closing_prices = np.array([df['close'].tolist() for df in list(self.values())], dtype=np.float32).T
        return closing_prices
