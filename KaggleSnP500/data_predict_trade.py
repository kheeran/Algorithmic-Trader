import numpy as np
import linecache # fastest when RAM is large enough to store the entire file (https://stackoverflow.com/questions/19189961/python-fastest-access-to-nth-line-in-huge-file). Otherwise use islice from itertools, which does not store file in RAM.
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# Simulate stream of daily prices (i.e. import cleaned data)
# It contains the fields ['Open', 'High', 'Low', 'Close', 'Volume'] for 10 (out of 503) stocks (names saved separately) for 757 days between 2019-01-01 and 2021-12-31.
class SimDataStream:
    def __init__(self, cleaned_data_csv, ticker_names_csv):

        # Initialise variables
        # 
        self.cleaned_data_csv = cleaned_data_csv # filename with daily streaming data
        self.ticker_names = np.genfromtxt(ticker_names_csv, delimiter=',', dtype=str) # stock tickers that the data corresponds to
        # 
        self.cur_linenum = 1 # index for index of current read line in csv
        self.cur_date = None  # current date of data corresponding to current line
        self.cur_values = None # row vector of today's values (data)
        self.at_end = False # boolean for end of stream

        self.col_names = linecache.getline(self.cleaned_data_csv, self.cur_linenum).rstrip('\n').split(',')[1:] # labels for each column/dimension of the row vector ('Date' is omitted)

        # check if next line is the end to indicate the end of the stream (i.e., the stream is empty)
        next_line = linecache.getline(self.cleaned_data_csv, self.cur_linenum + 1)
        if len(next_line) == 0:
            self.at_end = True


    # get vector of data/values for next trading day
    def nextTradingDay(self) -> tuple:
        
        # do nothing if stream has ended
        if self.at_end:
            print("Reached the end of the data stream.")
            return None

        # increment the trading day by incrementing line num in csv
        self.cur_linenum += 1

        # get the trading day's line of values in csv
        line_data = linecache.getline(self.cleaned_data_csv, self.cur_linenum).rstrip('\n').split(',')

        # set the trading day's date and vector/values appropriately
        self.cur_date = str(line_data[0])
        self.cur_values = [ float(val) for val in line_data[1:] ]

        # check if next line is the end of the stream
        next_line = linecache.getline(self.cleaned_data_csv, self.cur_linenum + 1)
        if len(next_line) == 0:
            self.at_end = True

        return self.cur_values


# Algorithmic Trader
class Trader:
    def __init__(self, init_cash : int, ticker_names : list, col_names : list, max_hist_len : int) -> None:

        # settings of trader
        self.num_days = 0 #  number of trading days
        self.cur_cash = init_cash # current available cash
        self.mytickers = ticker_names # stocks available to trade
        self.col_names = col_names # labels for each entry of the daily value array/vector
        self.max_hist_len = max_hist_len # maximum number of days of history allowed to store in memory

        # prices/volumes of stocks, and current shares held and the price they were bought for, and profits for each stock
        self.cur_vals = None
        self.hist_vals = []
        self.cur_shares_held = np.zeros(len(self.mytickers))
        self.cur_shares_held_price = np.zeros(len(self.mytickers))
        self.total_shares_profit = np.zeros(len(self.mytickers))

        # current buy/sell orders for stocks
        self.cur_buy_ords = [ None for _ in range(len(self.mytickers)) ]
        self.cur_sell_ords = [ None for _ in range(len(self.mytickers)) ]

        # prediction models
        self.cur_models = {}


        # sanity check
        # validate that ticker names and columns are exactly what is expected (so we can index instead of lookup, i.e., fast numpy arrays)
        fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        col_name_index = 0
        for ticker in self.mytickers:
            for field in fields:
                if f'{ticker}_{field}' == self.col_names[col_name_index]:
                    col_name_index += 1
                else:
                    raise Exception("Ordered column names do not match what is expected according to ordered ticker labels.")



    # initialise the day of trading and update order books
    def initDay(self, today_vals : list) -> None:
        self.num_days += 1
        self.cur_vals = today_vals
        self.updateOrders()
        self.updateHist()

        

    # update current order books
    # use today (end of day) values to evaluate
    # if buy/sell order is between the high and low, then order goes through (at 5% volume max for simplicity)
    # automatically cancel all orders that do not go through (simplicity)
    def updateOrders(self) -> None:

        for i in range(len(self.mytickers)):
            # Recall fields are ['Open', 'High', 'Low', 'Close', 'Volume']
            cur_high = self.cur_vals[i*5 + 1]
            cur_low = self.cur_vals[i*5 + 2]
            cur_vol = self.cur_vals[i*5 + 4]

            # update ticker's buy orders 
            if self.cur_buy_ords[i] != None:
                ord_price, ord_amount = self.cur_buy_ords[i]
                
                # Process order if price is good
                if cur_low <= ord_price <= cur_high:
                    ord_amount = min(ord_amount, 0.05*cur_vol) # allow at most 5% today's volume for simplicity
                    ord_value = ord_amount*ord_price

                    # Validate if enough cash to purchase and then process order
                    if self.cur_cash > ord_value:
                        self.cur_shares_held[i] += ord_amount
                        self.cur_cash -= ord_value

                        # update price of hare
                        self.cur_shares_held_price[i] = ord_price

                # remove corresponding buy order and cancel any remaining ones
                self.cur_buy_ords[i] = None

            # update ticker's sell orders
            if self.cur_sell_ords[i] != None:
                ord_price, ord_amount = self.cur_sell_ords[i]

                # Process order if price is good
                if cur_low <= ord_price <= cur_high:
                    ord_amount = min(ord_amount, 0.05*cur_vol) # allow at most 5% today's volume for simplicity
                    ord_value = ord_amount*ord_price

                    # validate if enough shares to sell and then process order
                    if self.cur_shares_held[i] >= ord_amount:
                        self.cur_shares_held[i] -= ord_amount
                        self.cur_cash += ord_value

                        # calc profit
                        profit = (ord_price - self.cur_shares_held_price[i])*ord_amount
                        
                        # update total profit
                        self.total_shares_profit[i] += profit

                # remove corresponding sell order and cancel any remaining ones
                self.cur_sell_ords[i] = None
    
    # update the stored history of values
    # NOTE: Could store the hist_vals as a custom queue that allows for constructing an array from the elements in the queue without removing them. For now, use slow list with O(n) dequeue.
    def updateHist(self) -> None:
        while len(self.hist_vals) >= self.max_hist_len:
            self.hist_vals.pop(0)
        self.hist_vals.append(self.cur_vals)


    # helper function for target label
    def helperTargLabel(self, col_name, d):
        return f'{col_name}_+{d}'

    # helper dataframe construction function
    def constructDataFrame(self, hist_vals, col_names, d):

        # Create dataframe from historical data 
        df = pd.DataFrame(hist_vals, columns = col_names)

        # setup observed columns for supervised learning
        obs_cols = [*col_names]
        targ_cols = []

        # construct relevant observed and target columns
        for col_name in col_names:

            # difference data for upto d diffs for each column (observed)
            for i in range(d):
                new_col = f'{col_name}_diff-{i+1}'
                df[new_col] = df[col_name].diff(i+1)
                obs_cols.append(new_col)
                
            # +d days future for each column (target)
            new_col = self.helperTargLabel(col_name, d)
            df[new_col] = df[col_name].shift(-d)
            targ_cols.append(new_col)

        return df, obs_cols, targ_cols
        
        
    # create models for each value being predicted and for +1 and +2 days (many models)
    # structure models as dictionary for easy access
    def trainModels(self,d) -> None:
        
        # helper func to construct relevant dataframe
        df, obs_cols, targ_cols = self.constructDataFrame(self.hist_vals, self.col_names, d)

        # remove rows with incomplete data
        df = df.dropna()

        # get observed variables
        Xs = df[obs_cols].values

        # construct prediction model for each target column (i.e., future values/data)
        for targ_name in targ_cols:
            
            # get target variable
            ys = df[targ_name]

            # train model
            model = RandomForestRegressor()

            model.fit(Xs, ys.ravel()) # ravel needed since single feature to be learned

            self.cur_models[targ_name] = model

    # evaluate each model and return vector of +d day predictions
    def getPreds(self, d) -> list:
        preds = []
        
        # get observations (only today's observed data is kept)
        df, obs_cols, _ = self.constructDataFrame(self.hist_vals[-(d + 1):], self.col_names, d)
        X = df[obs_cols].dropna().values # drop unnecessary rows (all but today)

        # get prediction for each column (i.e., vector dimension)
        for col_name in self.col_names:
            pred_name = self.helperTargLabel(col_name, d)
            if pred_name in self.cur_models:
                y_pred = self.cur_models[pred_name].predict(X)
            else:
                raise Exception(f"Predictive model does not exist for label {pred_name}.")
            preds.append(y_pred[-1])
        
        return preds

    # create new orders using predicted values for the subsequent days
    # simple: sell all stocks based on +1 day predictions and buy based on +1 and +2 day preictions
    def createOrders(self) -> None:

        # buy and sell price will be slightly price_offset (above and below, resp.) of the mid price, and volume to be a small fraction only
        price_offset = 0.25
        vol_frac = 0.01

        # store planed buy orders to spread capital accordingly (not needed for sell)
        exp_buy_orders = [ None for _ in range(len(self.cur_shares_held))]
        exp_total_buy_price = 0

        # get predictions
        pred_vals_1 = self.getPreds(1)
        pred_vals_2 = self.getPreds(2)

        # create buy and sell order for each stock (do not buy and sell the same stock for simplicity)
        for i in range(len(self.cur_shares_held)):

            # use relevant predictions for this stock 
            # (recall ['Open', 'High', 'Low', 'Close', 'Volume'])
            pred_high_1 = pred_vals_1[i*5 + 1]
            pred_low_1 = pred_vals_1[i*5 + 2]
            pred_vol_1 = pred_vals_1[i*5 + 4]
            pred_high_2 = pred_vals_2[i*5 + 1]
            pred_low_2 = pred_vals_2[i*5 + 2]
            pred_vol_2 = pred_vals_2[i*5 + 4]
            
            # sell if shares of stock held
            if self.cur_shares_held[i] > 0:
                # set sell order price below pred mid and set 1% of pred volume +1
                sell_price = pred_low_1 + (pred_high_1 - pred_low_1)*(0.5 + price_offset)
                ord_vol = min(self.cur_shares_held[i],pred_vol_1*vol_frac)
                
                # add to sell orders (+ sanity check)
                if self.cur_sell_ords[i] == None:
                    self.cur_sell_ords[i] = (sell_price, ord_vol)
            
            # buy if no shares of stock held and is predicted to sell for less the subsequent day
            elif self.cur_shares_held[i] == 0:

                # get expected buy price tomorrow and sell price the next day
                exp_buy_price = pred_low_1 + (pred_high_1 - pred_low_1)*(0.5 - price_offset)
                exp_sell_price = pred_low_2 + (pred_high_2 - pred_low_2)*(0.5 + price_offset)
                exp_vol = pred_vol_1*vol_frac

                # if profit is expected to be made, then add to planned buy orders (+ sanity check)
                if exp_buy_price < exp_sell_price and self.cur_buy_ords[i] == None:

                    exp_buy_orders[i] = (exp_buy_price, exp_vol)

                    exp_total_buy_price += exp_buy_price*exp_vol

        # scale down expected buy orders to available cash

        scale = min(self.cur_cash/exp_total_buy_price, 1) if exp_total_buy_price > 0 else 1
        for i in range(len(self.cur_buy_ords)):
            if exp_buy_orders[i] != None:
                self.cur_buy_ords[i] = (exp_buy_orders[i][0]*1, exp_buy_orders[i][1]*scale) # NOTE: wrap in int to take floor func of volume
    
    # execute a day of trading
    def execDay(self, today_vals : list) -> None:
        self.initDay(today_vals)

        # only trin model and start trading when enough data has been stored
        if self.num_days >= self.max_hist_len:
            
            # train model less regularly
            if self.num_days == self.max_hist_len or self.num_days % 10 == 0:
                self.trainModels(1)
                self.trainModels(2)
            
            # create sell orders
            self.createOrders()
        

if __name__ == "__main__":

    # initialise data stream (with cleaned data)
    samp_size = 10
    stream = SimDataStream(f'cleaned_{samp_size}.csv', f'cleaned_{samp_size}_tickers.csv')

    # initialise trader
    trader = Trader(100, stream.ticker_names, stream.col_names, 25)

    # cycle through data stream
    while not stream.at_end:
        today_vals = stream.nextTradingDay()
        trader.execDay(today_vals)
        print("buy:", trader.cur_buy_ords)
        print("sell:", trader.cur_sell_ords)
        print("shares held ( ):", trader.cur_shares_held)
        print("shares held (p):", trader.cur_shares_held_price)
        print("profit:", trader.total_shares_profit)
        print(trader.num_days, trader.cur_cash)
        print()






