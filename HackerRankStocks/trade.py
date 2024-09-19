
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# stream simulator
class streamSim:
    def __init__(self, data_dict : dict) -> None:
        self.stocks = list(data_dict.keys())
        self.prices = data_dict
        self.current_day = 0

    def getNextPrices(self) -> dict:
        pass
    
    # get recent n days of prices
    def getRecentPrices(self, days : int):
        pass


# streaming algorithm
class myTrader:
    def __init__(self, mystocks : list, init_cash : int, max_hist_len : int, analysis = False):

        # settings of trader
        self.mystocks = mystocks
        self.cur_cash = init_cash
        self.max_hist_len = max_hist_len
        self.analysis = analysis
            
        # key info store
        self.mymodels = {} # current set of regression models (for each stock)
        self.myprofit_total = {} # tracking the total profit of each stock so far
        # 
        self.cur_stocks_held = {} # each key is a stock and each value is a list of tuples (num, price). Can make it just a single tuple (instead of a list of tuples) 
        # 
        self.cur_day_count = 0
        self.yesterday_prices = None
        self.today_prices = None
        self.today_prices_diff = None
        self.tomorrow_preds = None
        self.today_trades = None

        # key data store
        init_dict = {}
        for stock in self.mystocks:
            init_dict[stock] = [ None ]
        self.stored_prices = copy.deepcopy(init_dict)
        self.stored_prices_diff = copy.deepcopy(init_dict)
        self.stored_prices_tomorrow = copy.deepcopy(init_dict)
        if self.analysis:
            self.mydays_daily = [ self.cur_day_count ]
            self.mysells_daily = copy.deepcopy(init_dict)
            self.myprofits_daily = copy.deepcopy(init_dict)
            self.mypredictions_daily = copy.deepcopy(init_dict)
            self.mybuys_daily = copy.deepcopy(init_dict)
            self.mycash_daily = [ self.cur_cash ] # at end of day

    # initialise day
    def initDay(self, today_prices) -> None:
        self.cur_day_count += 1
        self.yesterday_prices = self.today_prices
        self.today_prices = today_prices
        self.today_prices_diff = {}
        self.tomorrow_preds = None
        self.today_trades = None

        # set type of prices, check validity, and compute price difference
        for stock in self.mystocks:
            try:
                self.today_prices[stock] = float(self.today_prices[stock])
            except:
                raise TypeError("Number string contians invalid characters. Can not convert to float!")

            self.today_prices_diff[stock] = self.today_prices[stock] - self.yesterday_prices[stock]

        if self.analysis:
            self.mydays_daily.append(self.cur_day_count)

    # train pred models for each stock
    def trainModels(self) -> None:

        # create dataframe
        df = pd.DataFrame.from_dict(self.stored_prices).astype('float64')

        # add extra columns for training (observed and target)
        def next_day_label(stock):
            return f'{stock} (+1 day)'

        def day_diff_label(stock):
            return f'{stock} (diff)'

        obscols = self.mystocks.copy()
        targcols = []
        for stock in self.mystocks:
            ndlabel = next_day_label(stock)
            ddlabel = day_diff_label(stock)
            df[ndlabel] = df[stock].shift(-1)
            df[ddlabel] = df[stock].diff()
            obscols.append(ddlabel)
            targcols.append(ndlabel)

        # drop rows with at least one null entry
        df = df.dropna()

        # Set observed variables (same for each stock's predictor)
        X_train = df[obscols]
        for stock in self.mystocks:
            # target variable (different for each stock)
            y_train = df[targcols]

            # fit random forest regression model and store (for each stock)
            model = RandomForestRegressor()
            model.fit(X_train.values, y_train.values.ravel())
            self.mymodels[stock] = model

        # NOTE: Could possibly make a continuous update version with own implemention of decision trees. E.g., after every 50 days, create a new decision tree with the recent 50 day data and combine with previous decision trees (like random forests, but data is not randomised) -- older trees can possibly be down weighted. This would reduce amount of data stored. TODO: Think about correctness guarantees of this strategy.

    # update prices
    def processPrices(self) -> None:
        # do not care about max history length for now (TODO)
        for stock in self.mystocks:
            self.stored_prices[stock].append(self.today_prices[stock])

        
    def getPreds(self) -> dict:
        # construct target array, which must be in same order as training. NOTE: loops can be removed if using numpy arrays instead of dictionaries, i.e., concatenate the corresponding numpy arrays.
        x = []
        for stock in self.mystocks: 
            x.append(self.today_prices[stock])
        for stock in self.mystocks:
            x.append(self.today_prices_diff[stock])

        # get predictions for each stock
        for stock in self.mystocks:
            self.tomorrow_preds[stock] = self.mymodels[stock].predict(x) # TODO: Double check format of this output

            
    # always sell everything (simple for now)
    def makeSells(self) -> None:
        sells = copy.deepcopy(self.cur_stocks_held)
        for stock in self.mystocks: # NOTE: this for loop could be removed by using numpy arrays instead of dictionaries
            if self.analysis:
                self.mysells_daily[stock].append(self.cur_stocks_held[stock])
            
            self.cur_stocks_held[stock] = 0
        return sells


    def makeBuys(self, preds : dict) -> dict:
        # if buying today and selling tomorrow makes a profit (based on predictions), then buy today
        # amount to buy will depend on amount of profit made so far on a particular stock

        for stock in self.mystocks:
        HERE!!



    def makeTrades(self) -> dict:
        # actually update the tables based on the buy and sell amounts
        # and return trades
        pass

    def executeDay(self) -> dict:
        self.initDay()
        pass






    





