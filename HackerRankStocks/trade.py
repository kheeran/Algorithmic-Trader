
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns




# stream simulator
class streamSim:
    def __init__(
            self, 
            data_dict : dict
    ) -> None:
        self.all_stocks = list(data_dict.keys())
        self.all_prices = data_dict
        self.current_day = 0
        self.end_of_stream = False

        # check that data for each stock is in a list of arrays
        for stock in self.all_stocks:
            if type(self.all_prices[stock]) != list:
                raise TypeError("Prices are not presented as an array/list type!")


    def getNextDayPrices(self) -> dict:
        self.current_day += 1
        today_prices = {}
        for stock in self.all_stocks:
            # check to see if stock prices are available for that day
            if len(self.all_prices[stock]) > self.current_day:
                today_prices[stock] = self.all_prices[stock][self.current_day - 1]
        
        # check to see if it is the end of the data set, i.e., no stock prices left
        if not today_prices:
            self.end_of_stream = True
        return today_prices
    
    # get recent n days of prices -- not needed now, implement later (TODO)
    def getRecentPrices(self, days : int):
        pass


# streaming algorithm for trading
class myTrader:
    def __init__(
            self, 
            mystocks : list, 
            init_cash : int, 
            max_hist_len : int, 
            regular_update = False,
            analysis = False
    ):

        # settings of trader
        self.mystocks = mystocks
        self.max_hist_len = max_hist_len
        self.analysis = analysis
        self.regular_update = regular_update
            
        # key variables stored
        self.cur_cash = init_cash
        self.cur_stocks_held_value = 0
        self.cur_day_count = 0
        self.yesterday_prices = {}
        self.today_prices = {}
        self.today_prices_diff = {}
        self.tomorrow_prices_preds = {}
        self.today_sells = {}
        self.today_buys = {}
        self.mymodels = {} # current set of regression models (for each stock)
        self.myprofits_total = { stock : 0 for stock in self.mystocks } # tracking the total profit of each stock so far
        self.cur_stocks_held = { stock : 0 for stock in self.mystocks }  

        # key arrays stored (over many days, inclulding initial day 0 of no activity/initialisation)
        self.stored_prices = { stock : [] for stock in self.mystocks }
        self.stored_prices_diff = { stock : [] for stock in self.mystocks }
        self.stored_prices_tomorrow = { stock : [] for stock in self.mystocks }
        if self.analysis:
            self.mydays_daily = []
            self.mycash_daily = []
            self.mysells_daily = { stock : [] for stock in self.mystocks }
            self.myprofits_daily = { stock : [] for stock in self.mystocks }
            self.mypredictions_daily = { stock : [] for stock in self.mystocks }
            self.mybuys_daily = { stock : [] for stock in self.mystocks }
    
    # train prediction models for each stock
    def trainModels(self) -> None:

        # create dataframe
        df = pd.DataFrame.from_dict(self.stored_prices).astype('float64')

        # add extra columns for training (observed and target)
        def next_day_label(stock):
            return f'{stock} (+1 day)'

        def day_diff_label(stock):
            return f'{stock} (diff)'

        obscols = self.mystocks.copy()
        for stock in self.mystocks:
            ndlabel = next_day_label(stock)
            ddlabel = day_diff_label(stock)
            df[ndlabel] = df[stock].shift(-1)
            df[ddlabel] = df[stock].diff()
            obscols.append(ddlabel)

        # drop rows with at least one null entry
        df = df.dropna()

        # Set observed variables (same for each stock's predictor)
        X_train = df[obscols]
        for stock in self.mystocks:
            # target variable (different for each stock)
            y_train = df[next_day_label(stock)]

            # fit random forest regression model and store (for each stock)
            model = RandomForestRegressor()
            model.fit(X_train.values, y_train.values.ravel())
            self.mymodels[stock] = model

        # NOTE: Could possibly make a continuous update version with own implemention of decision trees. E.g., after every 50 days, create a new decision tree with the recent 50 day data and combine with previous decision trees (like random forests, but data is not randomised) -- older trees can possibly be down weighted. This would reduce amount of data stored. TODO: Think about correctness guarantees of this strategy.
        
    def getPreds(self) -> None:
        # construct observed data array, which must be in same order as training. NOTE: loops can be removed if using numpy arrays instead of dictionaries, i.e., concatenate the corresponding numpy arrays.
        x = [[]]
        for stock in self.mystocks: 
            x[0].append(self.today_prices[stock])
        for stock in self.mystocks:
            x[0].append(self.today_prices_diff[stock])

        # get predictions for each stock
        for stock in self.mystocks:
            if stock in self.mymodels.keys():
                self.tomorrow_prices_preds[stock] = self.mymodels[stock].predict(x)
            else:
                self.tomorrow_prices_preds[stock] = None

            if self.analysis:
                self.mypredictions_daily[stock].append(self.tomorrow_prices_preds[stock])

    def initDay(self, today_prices) -> None:
        self.cur_day_count += 1
        self.yesterday_prices = self.today_prices
        self.today_prices = today_prices
        self.today_prices_diff = {}


        to_delete = []
        # set type of prices, check validity, and compute price difference
        for stock in self.mystocks:
            try:
                if stock in self.today_prices.keys():
                    self.today_prices[stock] = float(self.today_prices[stock])
                else:
                    # if price is no longer available for today, nullify previous trade
                    self.today_prices[stock] = self.yesterday_prices[stock]
                    to_delete.append(stock)

                # no price difference for day 1
                if stock in self.yesterday_prices.keys():
                    self.today_prices_diff[stock] = self.today_prices[stock] - self.yesterday_prices[stock]
                else:
                    self.today_prices_diff[stock] = None
                    
            except:
                print(f"Failed at converting the stock price to float. Stock {stock} will be removed!")
                # NOTE: Handled so that all stocks will still later be sold.
                to_delete.append(stock)
        
        # cleanup deletion of keys
        for stock in to_delete:
            self.mystocks.remove(stock)

        # add to price history and truncate
        for stock in self.mystocks:
            self.stored_prices[stock].append(self.today_prices[stock])
            self.stored_prices[stock] = self.stored_prices[stock][-self.max_hist_len:]

        if self.analysis:
            self.mydays_daily.append(self.cur_day_count)
            self.mycash_daily.append(self.cur_cash)

        # reset some variables (for consistency)
        self.tomorrow_prices_preds.clear()
        self.today_sells.clear()
        self.today_buys.clear()

    # simple sell (for now): always sell everything
    def makeSells(self) -> None:

        for stock in self.cur_stocks_held.keys(): # NOTE: this for loop could be removed by using numpy arrays instead of dictionaries. NOTE: Using keys of self.cur_stocks_held instead of self.my_stocks to cater for when stocks are removed from consideration at any point.
            
            # sell all of this stock and update cash
            self.today_sells[stock] = self.cur_stocks_held[stock]
            self.cur_cash += self.today_sells[stock]*self.today_prices[stock]
            self.cur_stocks_held[stock] = 0

            # calculate profit for the selling of this stock (0 if no stock to sell)
            if self.today_sells[stock] > 0:
                profit_total = self.today_sells[stock]*self.today_prices_diff[stock]
            else:
                profit_total = 0

            self.myprofits_total[stock] += profit_total
  
            if self.analysis:
                self.mysells_daily[stock].append(self.today_sells[stock])
                self.myprofits_daily[stock].append(profit_total)

        # set value of stocks (always 0 after selling)
        self.cur_stocks_held_value = 0

    # simple buy (for now): only buy today if predicted to make a profit tomorrow and allow buying of stocks fractionally
    # amount to buy will depend on amount of profit made so far on a particular stock
    def makeBuys(self) -> None:
        # calculate separately for profit and loss making stocks
        frac_contrib_pos = np.zeros(len(self.mystocks))
        frac_contrib_neg = np.zeros(len(self.mystocks))
        for i, stock in enumerate(self.mystocks):
            # check if prediction exists (first check if key exists), else do not invest
            if stock in self.tomorrow_prices_preds.keys():
                if self.tomorrow_prices_preds[stock] != None:
                    # check if predicted price difference is positive, else do not invest
                    if self.tomorrow_prices_preds[stock] - self.today_prices[stock] > 0:
                        # add 1 so that trades are always made even if no profit has yet been made
                        frac_contrib_pos[i] += 1
                        frac_contrib_neg[i] += 1

                        # increment by total profit/loss so far (towards getting fraction of contributions)
                        if self.myprofits_total[stock] > 0:
                            frac_contrib_pos[i] += self.myprofits_total[stock]
                        else:
                            frac_contrib_neg[i] -= self.myprofits_total[stock]

        # calc fractional contributions (and handle divide by zero error)
        # NOTE: if no predictive model yet, then the frac_contrib arrays are all 0s and we do not buy anything
        pos_sum = frac_contrib_pos.sum()
        neg_sum = frac_contrib_neg.sum()
        if pos_sum != 0:
            frac_contrib_pos = frac_contrib_pos/pos_sum
        if neg_sum != 0:
            frac_contrib_neg = frac_contrib_neg/neg_sum

        # make buys appropriately
        cash_to_spend = self.cur_cash*0.999 # spend slightly less than total to cater for rounding errors
        total_cash_spent = 0 # sanity check, but should total to all spent
        for i, stock in enumerate(self.mystocks):
            # cash allocated and used for buying this stock
            # spare 10% of cash for loss making stocks (continually give a chance incase prediction improves over time)
            cash_allocated_stock = (frac_contrib_pos[i]*0.99 + frac_contrib_neg[i]*0.01)*(cash_to_spend)

            # amount of stocks to buy
            if self.today_prices[stock] <= 0:
                quantity_stock = 10000000 # just by alot if free or negative
            else:
                quantity_stock = int(cash_allocated_stock/self.today_prices[stock]) # take the floor
            self.today_buys[stock] = quantity_stock
            self.cur_stocks_held[stock] = quantity_stock

            total_cash_spent += quantity_stock*self.today_prices[stock] # recompute due to floor func
    
            if self.analysis:
                self.mybuys_daily[stock].append(quantity_stock)
            
        # sanity check buying
        self.cur_cash -= total_cash_spent
        self.cur_stocks_held_value = total_cash_spent
        if self.cur_cash < 0:
            raise Exception(f"Used too much money! Current cash is {self.cur_cash}")
            
        


    # the main function essentially
    def executeDay_old(self, today_prices : dict) -> dict:

        # standard day, only train model at 10th day
        self.initDay(today_prices)
        self.makeSells()
        if self.cur_day_count == self.max_hist_len:
            self.trainModels()
        self.getPreds()
        self.makeBuys()

        # structure sell buy output
        trades = {}
        for stock in self.mystocks:
            trades[stock] = (self.today_sells[stock], self.today_buys[stock])
        
        return trades

    # structure output
    def getTrades(self):
        trades={}
        for stock in self.mystocks:
            if bool(self.today_buys):
                trades[stock] = f'{stock} BUY {self.today_buys[stock]}'
            else:
                trades[stock] = f'{stock} SELL {self.today_sells[stock]}'
        return trades

    # processing a day
    def executeDay(self, today_prices : dict) -> dict:

        # initialise dat
        self.initDay(today_prices)
        
        if self.cur_day_count % 2 ==0:
            self.makeSells() # sell all

        # only train model after collected enough data
        if self.regular_update:
            cond = self.cur_day_count % self.max_hist_len*2 == 0
        else:
            cond = self.cur_day_count == self.max_hist_len
        if cond:
            self.trainModels()

        if self.cur_day_count % 2 == 1:
            self.getPreds()
            self.makeBuys()

        # returned appropriately structured output        
        return self.getTrades()
    
    # analyse trading period so far
    def analyseProfits(self):
        df = pd.DataFrame.from_dict(self.myprofits_daily).astype('float64')
        # df['daily_profits'] = df.sum(axis=1)
        df['cumulative_profits'] = df.sum(axis=1).cumsum(axis=0)
        
        df.insert(0, "day", [ i + 1 for i in range(int(self.cur_day_count/2))])

        # compare all stocks and total profit
        dfmelt = pd.melt(df, ['day'])

        # check days of loss
        # dfmelt = pd.melt(df[['total_profits', 'day']].loc[df['total_profits'] < 0], ['day'])

        plt.figure()
        plt.title("Daily Profits")
        sns.lineplot(data=dfmelt, x='day', y='value', hue='variable')
        plt.yscale('log')
        plt.show()


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
    trader = myTrader(stream.all_stocks, 100, 10, analysis = True)

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
