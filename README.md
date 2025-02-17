# Algorithmic Trader
#### Using Random Forests to exploit the correlation between the current prices/volume of stocks and their future prices/volumes. Backtesting on S&P500 stock data in the years 2019-2021.
----
##### 1. Data Cleaning ([/KaggleSnP500/data_clean.py](/KaggleSnP500/data_clean.py)):
  - Downloaded daily data for stocks from the S&P500 (https://www.kaggle.com/datasets/rprkh15/sp500-stock-prices) -- each stock corresponds to a single table of the fields [open, high, low, close, volume] for trading days going back many years from mid 2022 (number of years varies for each stock).
  - Randomly subsampled 100 to work with and aggregated all the information into a single table where rows represent trading days, keping only the rows where all fields were filled between the years 2019-2021.
##### 2. Data Exploration ([/KaggleSnP500/data_clean.py](/KaggleSnP500/data_explore.py)):
  - Verified the assumption that the features of stocks today are correlated to its features tomorrow.
  - Obtained the Pearson correlation coefficient for all pairwise features of a trading day (i.e., the fields [open, high, low, close, volume] of each stock) and plotted a heatmap of the coefficients to visualy inspect the correlations (many were strongly correlated).
##### 3. Training the Random Forests:
  - See [/Others/HackerRankStocks/predict.py](/Others/HackerRankStocks/predict.py) for a comparison of different feature extraction for the training of a RandomForest and their validation using the R2 score.
##### 4. Implementation ([/KaggleSnP500/data_predict_trade.py](/KaggleSnP500/data_predict_trade.py)):
  - A simple `SimDataStream` class to simulate trading day features (fields of each stock) arriving daily in chronological order.
  - A `Trader` class that, starting with an initial cash amount, does the following:
    - processes the daily features;
    - stores the daily features upto a maximum memory (number of days of history);
    - trains a Random Forest model when sufficient data has been stored; and
    - uses the model to make predictions that are used in the *simple trading strategy* as follows:
      - create buy orders for a certain stock if buying tomorrow and selling the next day makes a profit (based on the predictions), and
      - create sell orders to sell all shares held.
