import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import argparse
import os
from preprocessing.dataset_preprocessing import preprocess_dataset


def get_yfinance_data(short_name, start_date, end_date, freq):
    # Download the ae
    ticker = yf.Ticker(short_name)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    dataset = ticker.history(period=freq, start=start_date, end=end_date)[features]
    dataset.rename(columns={col: col.lower() for col in features}, inplace=True)
    dataset['date'] = dataset.index
    dataset['date'] = [str(date).split(' ')[0] for date in dataset['date']]
    #dataset['date'] = pd.Series([str(date)+' 00:00:00' for date in dataset['date']])

    # Preprocess ae
    preprocessed_dataset = preprocess_dataset(dataset)
    return preprocessed_dataset


def save_dataset(df, folder, filename):
    filepath = os.path.join(folder, filename) + '.csv'
    df.to_csv(filepath)


def parse_arguments():
    dataset_version = 'dataset_v2'
    parser = argparse.ArgumentParser(description='Get the ae from yfinance.')
    parser.add_argument('-short_name', '-short_name', default='orcl', type=str,
                        help='Short name of the stock for which you want to get the prices.')
    parser.add_argument('-start_date', '-start_date', default='2002-01-01', type=str,
                        help='Starting date of prices.')
    parser.add_argument('-end_date', '-end_date', default='2020-12-31', type=str,
                        help='Ending date of prices.')
    parser.add_argument('-freq', '-freq', default='1d', type=str,
                        help='Frequency of price ae.')
    parser.add_argument('-save_directory', '-save_directory', default=f'../../Datasets/{dataset_version}/1_preprocessed_data', type=str,
                        help='Directory to save the ae.')
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    dataset = get_yfinance_data(args.short_name, args.start_date, args.end_date, args.freq)
    save_dataset(dataset, args.save_directory, args.short_name)


if __name__ == '__main__':
    main()
