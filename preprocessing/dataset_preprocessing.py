from finta import TA
import numpy as np
from preprocessing.indicators import demand_index, commodity_selection_index, dynamic_momentum_index

def preprocess_dataset(dataset):
    # Turn columns into floats
    float_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in float_columns:
        dataset[[col]] = dataset[[col]].astype(float)

    # Drop missing values
    dataset = dataset.dropna()
    dataset.reset_index(drop=True, inplace=True)

    # Get ohlc and ohlcv
    ohlc = dataset[["open", "high", "low", "close"]]
    ohlcv = dataset[["open", "high", "low", "close", "volume"]]

    # Calculate indicators
    dataset["ema"] = TA.EMA(ohlc)
    dataset["hma"] = TA.HMA(ohlc)
    dataset["mom"] = TA.MOM(ohlc)
    dataset["atr"] = TA.ATR(ohlc)
    dataset["cci"] = TA.CCI(ohlc)
    dataset['dei'] = demand_index(dataset)
    dataset['csi'] = commodity_selection_index(dataset)
    dataset['dmi'] = dynamic_momentum_index(dataset)

    # Drop rows with NaNs
    dataset = dataset.dropna()
    dataset.reset_index(drop=True, inplace=True)

    # Normalize prices
    dataset["norm_close"] = dataset.apply(lambda row: row.close / row.open, axis=1)
    dataset["norm_low"] = dataset.apply(lambda row: row.low / row.open, axis=1)
    dataset["norm_high"] = dataset.apply(lambda row: row.high / row.open, axis=1)

    # Normalize indicators
    indicators = ['atr', 'cci', 'csi', 'dei', 'dmi', 'ema', 'hma', 'mom']
    norm_indicators = ["norm_" + ind for ind in indicators]
    dataset[norm_indicators] = dataset[indicators] / dataset[indicators].shift()
    dataset.loc[0, norm_indicators] = 1

    # Drop NaNs
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset = dataset.dropna()
    dataset.reset_index(drop=True, inplace=True)

    return dataset

def preprocess_dataset2(dataset):
    # Turn columns into floats
    float_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in float_columns:
        dataset[[col]] = dataset[[col]].astype(float)

    # Drop missing values
    dataset = dataset.dropna()
    dataset.reset_index(drop=True, inplace=True)

    # Normalize prices
    dataset["ae1"] = dataset.apply(lambda row: row.close / row.open, axis=1)
    dataset["ae2"] = dataset.apply(lambda row: row.low / row.open, axis=1)
    dataset["ae3"] = dataset.apply(lambda row: row.high / row.open, axis=1)

    # Drop NaNs
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset = dataset.dropna()
    dataset.reset_index(drop=True, inplace=True)

    return dataset

