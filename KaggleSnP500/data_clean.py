import pandas as pd
import os
import random
import datetime

if __name__ == '__main__':

    # consider only dates in 2019,2020,2021
    dates = []
    start_date = datetime.date(2019, 1, 1) 
    end_date = datetime.date(2021, 12, 31)    # perhaps date.today()
    delta = end_date - start_date   # returns timedelta
    for i in range(delta.days + 1):
        dates.append(str(start_date + datetime.timedelta(days=i)))

    # dicts for open price 
    prices = {}
    for root, _, files in os.walk('./archive'):
        # randomly sample ~100 stocks to work with (fix local randomness with seed)
        random.Random(125).shuffle(files)

        for file in files[:100]:
            df = pd.read_csv(os.path.join(root, file))
            stock = file[:3]

            # extract open, high, low, close, and volume (just open price for now) 
            df = df.loc[df['Date'] >= str(start_date)]
            df = df.loc[df['Date'] <= str(end_date)]

            # set up local columns
            this_dates = df['Date'].values
            this_opens = df['Open'].values
            # this_highs = df['High'].values
            # this_lows = df['Low'].values
            # this_closes = df['Close'].values
            # this_volumes = df['Volume'].values

            # prep stock in dictionary
            prices[stock] = []

            # make dates line up and fill spaces with None
            i = 0
            for j in range(len(this_dates)):
                if dates[i] == this_dates[j]:
                    prices[stock].append(this_opens[j])
                    i += 1
                else:
                    while dates[i] != this_dates[j]:
                        prices[stock].append(None)
                        i += 1
                    prices[stock].append(this_opens[j])
                    i += 1

            

    for key in prices.keys():
        print(prices[key])