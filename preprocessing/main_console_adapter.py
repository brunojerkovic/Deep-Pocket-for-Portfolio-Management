from preprocessing.extract_excel import get_stocks, extract_excel
from preprocessing.stream_yfinance import get_yfinance_data, save_dataset
import os


def preprocess_all_data(args, experiment_name):
    # Create a folder to store the preprocessed ae
    os.makedirs(args.save_directory, exist_ok=True)

    # Extract the excel sheet with stocks
    short_names, long_names = get_stocks(args)
    extract_excel(short_names, long_names, args.excel_filepath, args.save_directory)

    # Prepare the data to be downloaded from Yahoo's database
    yahoo_start_date, yahoo_end_date = '2002-01-01', '2020-12-31'
    yahoo_freq = '1d'
    preprocessed_stocks = [file.split('.csv')[0] for file in os.listdir(args.save_directory)]
    stocks_to_download = [stock for stock in short_names if stock not in preprocessed_stocks]

    # Download from Yahoo ae that is not available in Excel
    for short_name in stocks_to_download:
        dataset = get_yfinance_data(short_name, yahoo_start_date, yahoo_end_date, yahoo_freq)
        save_dataset(dataset, args.save_directory, short_name)
        print(f"Downloaded {short_name} from Yahoo!")
