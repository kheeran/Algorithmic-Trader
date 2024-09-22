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

    # dicts for open, high, low, close, and volume 
    stocks_info = {}

    # define main columns and helper label function
    main_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    mylabel = lambda a,b : f'{a}_{b}'


    for root, _, files in os.walk('./archive'):
        # randomly sample ~100 stocks to work with (fix local randomness with seed)
        random.Random(125).shuffle(files)

        for file in files[:10]:
            df = pd.read_csv(os.path.join(root, file))
            stock = file[:3]

            # retain rows with relevant dates
            df = df.loc[df['Date'] >= str(start_date)]
            df = df.loc[df['Date'] <= str(end_date)]
            stock_dates = df['Date'].values

            # get values of relevant columns
            stock_info = df[main_cols].values

            # prep relevant arrays in dictionaries
            for key in main_cols:
                stocks_info[mylabel(stock,key)] = []

            # make dates line up (without any gaps) and fill spaces with None
            i = 0
            for j in range(len(stock_dates)):
                if dates[i] == stock_dates[j]:
                    for k, key in enumerate(main_cols):
                        stocks_info[mylabel(stock,key)].append(stock_info[j][k])
                    i += 1
                else:
                    while dates[i] != stock_dates[j]:
                        for key in main_cols:
                            stocks_info[mylabel(stock,key)].append(None)
                        i += 1
                    for k, key in enumerate(main_cols):
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

    print(main_df.head())



