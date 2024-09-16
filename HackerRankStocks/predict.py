import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# read raw data
raw_data = pd.read_csv('~/Documents/PythonRefresher/HackerRankStocks/traindata.txt', header=None)

# clean data
data_dict = {}
max_len = 0
for line in raw_data[0]:
    details = line.strip().split()
    stock_name = details[0]
    stock_prices = details[1:]
    data_dict[stock_name] = stock_prices
    if len(stock_prices) > max_len:
        max_len = len(stock_prices)

# create data frame
maindf = pd.DataFrame.from_dict(data_dict).astype('float64')
maindf.head()
cols = list(data_dict.keys())

# useful label function
def next_day_label(col):
    return f'{col} (next day)'

# add next day features
for col in cols:
    new_name = next_day_label(col)
    maindf[new_name] = maindf[col].shift(-1)

corr = maindf.dropna().corr()

plt.figure()
sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={'size': 5})
plt.show()


# split data into train and test
# train_data, test_data = train_test_split()


# construct random forrest for each stock
# for col in cols:


    
    


