
from tradelib import streamSim, myTrader
import pandas as pd
import time


# simulating a daily trading market given some data 
if __name__ == '__main__':

    # read raw data
    raw_data = pd.read_csv('~/Documents/PythonRefresher/HackerRankStocks/traindata.txt', header=None)

    # structure dictionary
    data_dict = {}
    for line in raw_data[0]:
        details = line.strip().split()
        stock_name = details[0]
        stock_prices = details[1:]
        data_dict[stock_name] = stock_prices

    # initialise stream simulator and trader
    stream = streamSim(data_dict)
    trader = myTrader(stream.all_stocks, 100, 5, analysis = True)

    start = time.time()
    while not stream.end_of_stream:
        today_prices = stream.getNextDayPrices()

        trades = trader.executeDay(today_prices)

        # # can improve speed by using numpy arrays instead of looping over the keys of dicts for everything
        if trader.cur_day_count % 25 == 0:    
            print()
            print(trader.cur_day_count, trader.cur_cash, trader.cur_stocks_held_value)
            print(trader.myprofits_total)


    time_taken = time.time() - start
    print()
    print("num of days trading:", trader.cur_day_count)
    print("total cash:", trader.cur_cash)
    print("profits:", trader.myprofits_total)
    print("total time (s):", round(time_taken, 3))
    print("average time per trade (s):", round(time_taken/trader.cur_day_count, 3))

    trader.analyseProfits()
