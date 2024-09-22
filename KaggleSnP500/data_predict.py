import pandas as pd
import numpy as np

# import cleaned data
# It contains the fields ['Open', 'High', 'Low', 'Close', 'Volume'] for 10 (out of 503) stocks (names saved separately) for 757 days between 2019-01-01 and 2021-12-31.
main_df = pd.read_csv("cleaned_10.csv")
tickers = np.genfromtxt("cleaned_10_tickers.csv", delimiter=',', dtype=str)


# TODO: construct training data so that it is trained on days where trading occurs on the day before AND the day after. See if this changes anything.

# create columns for difference between trading days and for next trading day prices (assuming no unusual gaps in the data for trading days)

# create models for day+1 and day +1, high low volume (6 predictors per stock), and for 20 stocks to trade with. Train on open, close, high, low, and volume

