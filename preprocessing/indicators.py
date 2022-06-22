from finta import TA
import numpy as np
import pandas as pd
import os


def commodity_selection_index(df):
    start_ind = 15
    csi = np.empty(len(df))

    atr = np.array(TA.ATR(df[["open", "high", "low", "close"]]))
    adx = np.array(TA.ADX(df[["open", "high", "low", "close"]]))

    for i in range(0, len(df)):
        if i < start_ind:
            csi[i] = np.nan
        else:
            adxr = (adx[i] + adx[i - 14]) / 2
            csi[i] = adxr * atr[i]

    return pd.Series(csi)


def dynamic_momentum_index(df, verbose=False):
    start_ind_ma = 10 - 1
    start_ind_std = 5 - 1

    close = np.array(df[['close']])
    std = np.array(
        [np.std(close[i - start_ind_std:i + 1]) if i >= start_ind_std else np.nan for i in range(0, len(close))])

    rsi = []
    for i in range(0, len(close)):
        if i < start_ind_ma + start_ind_std:
            rsi.append(np.nan)
        else:
            std_a = np.mean(std[i - start_ind_std:i + 1])
            v_i = std[i] / std_a
            td = int(14 / v_i)
            td = 5 if td < 5 else 30 if td > 30 else td
            td = i if i < td else td  # For border cases in the beginning, you can only go in history as much as you have it

            ohlc = df.loc[(i - td + 1):i + 1, ['open', 'high', 'low', 'close']]
            current_rsi = np.array(TA.RSI(ohlc=ohlc, period=td))[-1]
            rsi.append(current_rsi)

        if verbose:
            os.system('CLS')# clear_output(wait=True)
            print(f"Completed: {round(i / len(df) * 100)}%")

    return pd.Series(rsi)


def demand_index(df):
    def volatility_average(highs, lows):
        s = 0
        for i in range(1, len(highs)):
            h = max(highs[i], highs[i - 1])
            l = min(lows[i], lows[i - 1])
            s += (h - l)

        s /= (len(highs) - 1)
        return s

    di = np.zeros(len(df))
    di[0:10] = np.nan

    for i in range(10, len(df)):
        ohlcv = {
            'o': df.loc[i, 'open'],
            'h': df.loc[i, 'high'],
            'l': df.loc[i, 'low'],
            'c': df.loc[i, 'close'],
            'v': df.loc[i, 'volume']
        }

        highs = np.array(df.loc[i - 10:i, 'high'])
        lows = np.array(df.loc[i - 10:i, 'low'])
        VA = volatility_average(highs, lows)

        K = 3 * ohlcv['c'] / VA

        if ohlcv['c'] > ohlcv['o']:
            # Prices rise
            P = ohlcv['c'] / ohlcv['o']
            P *= K

            SP = ohlcv['v'] / P
            BP = ohlcv['v']
        else:
            # Prices decline
            P = ohlcv['c'] / ohlcv['o']
            P *= K

            SP = ohlcv['v']
            BP = ohlcv['v'] / P

        di[i] = BP / SP

    return pd.Series(di)