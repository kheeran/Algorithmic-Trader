
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


# streaming algorithm for trading
class myTrader:
    def __init__(self, mystocks : list, init_cash : int, max_hist_len : int, analysis = False):

        # settings of trader
        self.mystocks = mystocks
        self.cur_cash = init_cash
        self.max_hist_len = max_hist_len
        self.analysis = analysis
            
        # key variables stored
        self.cur_day_count = 0
        self.yesterday_prices = {}
        self.today_prices = {}
        self.today_prices_diff = {}
        self.tomorrow_preds = {}
        self.today_sells = {}
        self.today_buys = {}
        self.mymodels = {} # current set of regression models (for each stock)
        self.myprofit_total = { stock : 0 for stock in self.mystocks } # tracking the total profit of each stock so far
        self.cur_stocks_held = { stock : 0 for stock in self.mystocks }  

        # key arrays stored (over many days, inclulding initial day 0 of no activity/initialisation)
        init_dict = { stock : [ None ] for stock in self.mystocks }
        self.stored_prices = { stock : [ None ] for stock in self.mystocks }
        self.stored_prices_diff = { stock : [ None ] for stock in self.mystocks }
        self.stored_prices_tomorrow = { stock : [ None ] for stock in self.mystocks }
        if self.analysis:
            self.mydays_daily = [ self.cur_day_count ]
            self.mysells_daily = { stock : [ None ] for stock in self.mystocks }
            self.myprofits_daily = { stock : [ None ] for stock in self.mystocks }
            self.mypredictions_daily = { stock : [ None ] for stock in self.mystocks }
            self.mybuys_daily = { stock : [ None ] for stock in self.mystocks }
            self.mycash_daily = [ self.cur_cash ] # at end of day

    


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
        
    def getPreds(self) -> None:
        # construct target array, which must be in same order as training. NOTE: loops can be removed if using numpy arrays instead of dictionaries, i.e., concatenate the corresponding numpy arrays.
        x = []
        for stock in self.mystocks: 
            x.append(self.today_prices[stock])
        for stock in self.mystocks:
            x.append(self.today_prices_diff[stock])

        # get predictions for each stock
        for stock in self.mystocks:
            try: # added this try catch just in case, but probably not needed (i.e., handled in execute day function)
                self.tomorrow_preds[stock] = self.mymodels[stock].predict(x) # TODO: Double check format of this output
            except:
                self.tomorrow_preds[stock] = None

    # initialise day
    def initDay(self, today_prices) -> None:
        self.cur_day_count += 1
        self.yesterday_prices = self.today_prices
        self.today_prices = today_prices
        self.today_prices_diff = {}

        # set type of prices, check validity, and compute price difference
        for stock in self.mystocks:
            try:
                self.today_prices[stock] = float(self.today_prices[stock])
                self.today_prices_diff[stock] = self.today_prices[stock] - self.yesterday_prices[stock]
            except:
                print(f"Stock price contians an invalid character(s) and can not convert to float. Removing stock {stock} entirely!")
                # NOTE: Handled so that all stocks will still be sold.

        # add to price history (do not care about max history length for now -- TODO)
        for stock in self.mystocks:
            self.stored_prices[stock].append(self.today_prices[stock])

        if self.analysis:
            self.mydays_daily.append(self.cur_day_count)

        # reset some variables (for consistency)
        self.tomorrow_preds = {}
        self.today_sells = {}
        self.today_buys = {}

    # simple (for now): always sell everything
    def makeSells(self) -> None:
        self.today_sells = copy.deepcopy(self.cur_stocks_held)
        for stock in self.today_sells.keys(): # NOTE: this for loop could be removed by using numpy arrays instead of dictionaries. NOTE: Using keys of self.today_sells instead of self.my_stocks to cater for the unlikely glitch of price containing an invalid character, i.e., if this occurs, all held stocks of the err stock are still sold.

            # calculate profit for this stock
            profit_total = self.today_sells[stock]*self.today_prices_diff[stock]

            # HERE!!

            if self.analysis:
                self.mysells_daily[stock].append(self.cur_stocks_held[stock])
                self.myprofits_daily[stock].append(profit_total)
            
            self.cur_stocks_held[stock] = 0



    def makeBuys(self) -> None:
        # simple (for now): only buy today if predicted to make a profit tomorrow
        # amount to buy will depend on amount of profit made so far on a particular stock
        pass





    def makeTrades(self) -> dict:
        # actually update the tables based on the buy and sell amounts
        # and return trades
        pass


    # the main function essentially
    def executeDay(self) -> dict:
        self.initDay()

        # check if model exists before making predictions
        pass






    





