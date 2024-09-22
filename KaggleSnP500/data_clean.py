import pandas as pd
import os
import random
import datetime
import numpy as np

if __name__ == '__main__':

    # consider only dates in 2019,2020,2021
    dates = []
    start_date = datetime.date(2019, 1, 1) 
    end_date = datetime.date(2021, 12, 31) 
    delta = end_date - start_date   # returns timedelta
    for i in range(delta.days + 1):
        dates.append(str(start_date + datetime.timedelta(days=i)))

    # subsample to ~100 stocks
    total_size = 0 # filled later
    samp_size = 101
    samp_stocks = []

    # dicts for open, high, low, close, and volume 
    stocks_info = {}

    # define main columns and helper label function
    main_info = ['Open', 'High', 'Low', 'Close', 'Volume']
    mylabel = lambda a,b : f'{a}_{b}'


    for root, _, files in os.walk('./archive'):
        # randomly sample ~100 stocks to work with (fix local randomness with seed)
        total_size = len(files)
        random.Random(125).shuffle(files)

        for file in files[:min(samp_size, total_size)]:
            df = pd.read_csv(os.path.join(root, file))

            # save stocks sampled
            stock = file[:-4]
            samp_stocks.append(stock)

            # retain rows with relevant dates
            df = df.loc[df['Date'] >= str(start_date)]
            df = df.loc[df['Date'] <= str(end_date)]
            stock_dates = df['Date'].values

            # get values of relevant columns
            stock_info = df[main_info].values

            # prep relevant arrays in dictionaries
            for key in main_info:
                stocks_info[mylabel(stock,key)] = []

            # make dates line up (without any gaps) and fill spaces with None
            i = 0
            for j in range(len(stock_dates)):
                if dates[i] == stock_dates[j]:
                    for k, key in enumerate(main_info):
                        stocks_info[mylabel(stock,key)].append(stock_info[j][k])
                    i += 1
                else:
                    while dates[i] != stock_dates[j]:
                        for key in main_info:
                            stocks_info[mylabel(stock,key)].append(None)
                        i += 1
                    for k, key in enumerate(main_info):
                        stocks_info[mylabel(stock,key)].append(stock_info[j][k])
                    i += 1

    # construct main dataframe
    main_df = pd.DataFrame.from_dict(stocks_info)
    main_df.insert(0, 'Date', dates)
    # print(main_df.head())

    # only keep columns with more non-Nan entries than its median and then remove rows with any NaN entry
    non_na_sum = main_df.notna().sum()
    med = non_na_sum.median()
    good_cols = non_na_sum.loc[non_na_sum >= med].index
    main_df = main_df[good_cols].dropna()

    # check that all remaining sampled stocks have open, high, low, close, and volume (after clearing NaNs)
    final_cols = set(main_df.columns)
    final_stocks = []
    for stock in samp_stocks:
        sum_check = 0
        for key in main_info:
            if mylabel(stock, key) in final_cols:
                sum_check += 1
        if sum_check == len(main_info):
            final_stocks.append(stock)
        elif sum_check != 0:
            raise Exception("Stock {stock} does not have all the relevant fields.")
        
    # export dataframe and final stocks to csv
    main_df.to_csv(f"cleaned_{len(final_stocks)}.csv", index=False)

    np.savetxt(f"cleaned_{len(final_stocks)}_tickers.csv", np.array(final_stocks), delimiter=',', fmt='%s')

    print(f"Table saved! It contains the fields {main_info} for {len(final_stocks)} (out of {total_size}) stocks (names saved separately) for {main_df.shape[0]} days between {start_date} and {end_date}.")

    


