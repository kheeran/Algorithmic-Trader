import numpy as np
import linecache # fastest when RAM is large enough to store the entire file (https://stackoverflow.com/questions/19189961/python-fastest-access-to-nth-line-in-huge-file). Otherwise use islice from itertools, which does not store file in RAM.

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

        self.columns = linecache.getline(self.cleaned_data_csv, self.cur_linenum).rstrip('\n').split(',')[1:] # labels for each column/dimension of the row vector ('Date' is omitted)

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

        # get the trading day's line in csv
        cur_line = linecache.getline(self.cleaned_data_csv, self.cur_linenum).rstrip('\n').split(',')

        # set the trading day's date and vector/values appropriately
        self.cur_date = cur_line[0]
        self.cur_values = cur_line[1:]

        # check if next line is the end of the stream
        next_line = linecache.getline(self.cleaned_data_csv, self.cur_linenum + 1)
        if len(next_line) == 0:
            self.at_end = True



# Algorithmic Trader (TODO)
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
        self.cur_shares_prev_price = np.zeros(len(self.mytickers))
        self.total_shares_profit = np.zeros(len(self.mytickers))

        # prediction models
        self.cur_models_1 = [None for _ in range(len(self.col_names))]
        self.cur_models_2 = [None for _ in range(len(self.col_names))]

        # current buy/sell orders for stocks
        self.cur_buy_ords = {}
        self.cur_sell_ords = {}

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

        for i,ticker in enumerate(self.mytickers):
            cur_high = self.cur_vals[i*4 + 1]
            cur_low = self.cur_vals[i*4 + 2]
            cur_vol = self.cur_vals[i*4 + 4]

            # update ticker's buy orders 
            if ticker in self.cur_buy_ords:
                ord_price, ord_amount = self.cur_buy_ords[ticker]
                
                # Process order if price is good
                if cur_low <= ord_price <= cur_high:
                    ord_amount = min(ord_amount, 0.05*cur_vol) # allow at most 5% today's volume for simplicity
                    ord_value = ord_amount*ord_price

                    # Validate if enough cash to purchase and then process order
                    if self.cur_cash > ord_value:
                        self.cur_shares_held[i] += ord_amount
                        self.cur_cash -= ord_value

                    # remove corresponding buy order and cancel any remaining ones
                    del self.cur_buy_ords[ticker]

            # update ticker's sell orders
            if ticker in self.cur_sell_ords:
                ord_price, ord_amount = self.cur_sell_ords[ticker]

                # Process order if price is good
                if cur_low <= ord_price <= cur_high:
                    ord_amount = min(ord_amount, 0.05*cur_vol) # allow at most 5% today's volume for simplicity
                    ord_value = ord_amount*ord_price

                    # validate if enough shares to sell and then process order
                    if self.cur_shares_held >= ord_amount:
                        self.cur_shares_held[i] -= ord_amount
                        self.cur_cash += ord_value

                    # remove corresponding sell order and cancel any remaining ones
                    del self.cur_sell_ords[ticker]
    
    # update the stored history of values
    # NOTE: Could store the hist_vals as a custom queue that allows for constructing an array from the elements in the queue without removing them. For now, use slow list with O(n) dequeue.
    def updateHist(self) -> None:
        while len(self.hist_vals) >= self.max_hist_len:
            self.hist_vals.pop(0)
        self.hist_vals.append(self.cur_vals)

    # create models for each value being predicted and for +1 and +2 days (many models)
    # structure models as dictionary for easy access
    def trainModels(self) -> None:
        pass

    # evaluate each model and return a tuple of +1 day and +2 day predictions
    def predNextVals(self) -> tuple:
        pass

    # create new orders using predicted values for the subsequent days
    def createOrders(self, tomorrow_vals : list, day_after_vals : list) -> None:
        pass
    
    # execute a day of trading
    def execDay(self) -> None:
        self.initDay()
        pass




if __name__ == "__main__":

    # initialise data stream
    samp_size = 10
    stream = SimDataStream(f'cleaned_{samp_size}.csv', f'cleaned_{samp_size}_tickers.csv')

    # cycle through data stream
    while not stream.at_end:
        stream.nextTradingDay()
        print(stream.cur_date, stream.cur_values)



