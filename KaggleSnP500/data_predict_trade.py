import numpy as np
import linecache # fastest when RAM is large enough to store the entire file (https://stackoverflow.com/questions/19189961/python-fastest-access-to-nth-line-in-huge-file). Otherwise use islice from itertools, which does not store file in RAM.

# Simulate stream of daily prices (i.e. import cleaned data)
# It contains the fields ['Open', 'High', 'Low', 'Close', 'Volume'] for 10 (out of 503) stocks (names saved separately) for 757 days between 2019-01-01 and 2021-12-31.
class SimDataStream:
    def __init__(self, cleaned_data_csv, ticker_names_csv):

        # store cleaned data file name and list of ticker names
        self.cleaned_data_csv = cleaned_data_csv
        self.ticker_names = np.genfromtxt(ticker_names_csv, delimiter=',', dtype=str)

        # set current (first) line in csv, current date + values (i.e., None), and store tickers
        self.cur_linenum = 1
        self.cur_date = None
        self.cur_values = None
        self.at_end = False

        # readin columns (except 'Date') from cleaned csv
        self.columns = linecache.getline(self.cleaned_data_csv, self.cur_linenum).rstrip('\n').split(',')[1:]

        # check if next line is the end
        next_line = linecache.getline(self.cleaned_data_csv, self.cur_linenum + 1)
        if len(next_line) == 0:
            self.at_end = True


    def nextTradingDay(self):
        
        # do nothing if stream has ended
        if self.at_end:
            print("Reached the end of the data stream.")
            return None

        # increment the trading day by incrementing line num in csv
        self.cur_linenum += 1

        # get the trading day's line in csv
        cur_line = linecache.getline(self.cleaned_data_csv, self.cur_linenum).rstrip('\n').split(',')

        # set the trading day's date and values appropriately
        self.cur_date = cur_line[0]
        self.cur_values = cur_line[1:]

        # check if next line is the end
        next_line = linecache.getline(self.cleaned_data_csv, self.cur_linenum + 1)
        if len(next_line) == 0:
            self.at_end = True


# Algorithmic Trader (TODO)
class Trader:
    def __init__(self, init_cash : int, ticker_names : list, col_names : list, max_hist_len : int) -> None:

        # settings of trader
        self.cur_day = 0
        self.cur_cash = init_cash # current available cash
        self.mytickers = ticker_names # stocks to trade
        self.col_names = col_names # labels for each entry of the daily value array
        self.max_hist_len = max_hist_len

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
        self.cur_buy_ords = { ticker : None for ticker in self.mytickers }
        self.cur_sell_ords = { ticker : None for ticker in self.mytickers }

    # initialise the day of trading and update order books
    def initDay(self, today_vals : list) -> None:
        self.cur_day += 1
        self.cur_vals = today_vals
        self.updateOrders()
        self.updateHist()

        # sanity check
        # validate that ticker names and columns are exactly what is expected (so we can index instead of lookup)
        fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        # TODO

    # update current order books
    # use today (end of day) values to evaluate
    # if buy/sell order is between the high and low, then order goes through (at 5% volume max for simplicity)
    # automatically cancel all orders that do not go through (simplicity)
    def updateOrders(self) -> None:
        pass
    
    # update the stored history of values
    def updateHist(self) -> None:
        pass

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



