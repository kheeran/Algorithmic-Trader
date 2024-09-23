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
    def __init__(self):
        pass


if __name__ == "__main__":

    # initialise data stream
    samp_size = 10
    stream = SimDataStream(f'cleaned_{samp_size}.csv', f'cleaned_{samp_size}_tickers.csv')

    # cycle through data stream
    while not stream.at_end:
        stream.nextTradingDay()
        print(stream.cur_date, stream.cur_values)



