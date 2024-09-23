import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

## 1. import data and create data frame

# read raw data
raw_data = pd.read_csv('~/Documents/PythonRefresher/HackerRankStocks/traindata.txt', header=None)

# dicts for timeframes and raw data
dict_day = { 'day': [ i + 1 for i in range(len(raw_data[0][0].split())-1)] }
dict_data = {}
for line in raw_data[0]:
    line_array = line.split()
    label = line_array[0]
    prices = line_array[1:]
    dict_data[label] = prices

# Initialise
maindf = pd.DataFrame.from_dict(dict_data)
maindf = maindf.astype('float64')
maindf.insert(0, column='day', value=dict_day['day'])

# create n-day rolling average columns
n=10
cols = list(dict_data.keys())
cols_roll = [ f'{col}_{n}day_avg' for col in cols]
for i in range(len(cols)):
    maindf[cols_roll[i]] = maindf[cols[i]].rolling(window=n).mean()

# ## 2. exploring data

print(maindf.info()) # basic info
print()
print(maindf.isnull().sum()) # simple check
print()
print(maindf[cols].describe()) # basic stats

# pairwise correlation
corr = maindf[cols].corr()
plt.figure()
plt.title("Correlation of Stocks")
sns.heatmap(corr, annot=True, annot_kws={'size':9}, fmt=".2f")
plt.show()

# time series plot
meltdf = pd.melt(maindf[['day', *cols]], ['day'])
meltdf = meltdf.rename({'value': 'price', 'variable':'stock'}, axis=1)
plt.figure()
plt.title("Daily Stock Price")
sns.lineplot(data=meltdf, x='day', y='price', hue='stock')
plt.show()

# time series for rolling avg
meltdf = pd.melt(maindf[['day', *cols_roll]], ['day'])
meltdf = meltdf.rename({'value': 'price', 
'variable':'stock'}, axis=1)
plt.figure()
plt.title(f"{n}Day Rolling Avg Stock Price")
sns.lineplot(data=meltdf, x='day', y='price', hue='stock')
plt.show()

# compare daily price and n-day-avg price
lim = 3
meltdf = pd.melt(maindf[['day', *cols[:lim], *cols_roll[:lim]]], ['day'])
meltdf = meltdf.rename({'value': 'price', 
'variable':'stock'}, axis=1)
plt.figure()
plt.title(f"Daily Price vs {n}Day Rolling Avg Price")
sns.lineplot(data=meltdf, x='day', y='price', hue='stock')
plt.show()


# plot histogram of stock prices
plt.figure()
plt.title("Distribution of Prices")
plt.xlabel("price")
plt.ylabel("frequency")
sns.histplot(data=maindf[cols[:lim]], binwidth=2)
plt.show()











# # Time-Series Analysis Class
# class timeEDA:

#     ## Creating Data Frame

#     # initialise
#     def __init__(self, time, data):
#         self.len = len(list(time.values())[0])
#         self.time = list(time.keys())[0]
#         self.main_cols = list(data.keys())
#         self.cols = [self.time] + list(self.main_cols)
#         self.roll_avg_nums = set()

#         # create dataframe from dict
#         df = pd.DataFrame.from_dict(data)
#         df = df.astype('float64')
#         df.insert(0, self.time, time[self.time])
#         self.df = df

#     # rolling average column labels (helper func)
#     def get_cols_roll(self, num):
#         return [ f'{col}_{num}{self.time}_mean' for col in self.main_cols]
        

#     # create rolling average of each data column
#     def create_roll_avg(self, num):
        
#         # create new col labels
#         colsroll = self.get_cols_roll(num)

#         # add new cols to df
#         df = self.df
#         for i in range(len(colsroll)):

#             ## TODO: Throw (and handle) error if key already exists
#             df[colsroll[i]] = df[self.main_cols[i]].rolling(window=num).mean()
#             self.cols.append(colsroll[i])
#         self.df = df
#         self.roll_avg_nums.add(num)

#     # get rolling average dataframe
#     def get_roll_avg_df(self, num):
#         if num in self.roll_avg_nums:
#             return self.df[[self.time, *self.get_cols_roll(num)]]
#         else:
#             return self.df[self.time]

#     # helper function for verifying cols
#     def check_cols(self,cols):
#         if type(cols) == list:
#             n = len(cols)
#             cols = [col for col in cols if col in self.cols]
#             if n != len(cols):
#                 raise Exception("Error! Invalid column keys have been used.")
#             return cols
#         else:
#             raise TypeError("Only lists allowed.")


#     ## Exploratory Data Analysis

#     def print_general_info(self, cols=[]):
#         cols = self.check_cols(cols) 
#         if cols == []:
#             cols = self.cols      
#         df = self.df[cols] 
#         print('~~~~~~~~~~~~~~~~~~~~~~')
#         print("     General Info")
#         print('~~~~~~~~~~~~~~~~~~~~~~')
#         print(df.info())
#         print()
#         print('~~~~~~~~~~~~~~~~~~~~~~')
#         print("Columns with null/NaN")
#         print('~~~~~~~~~~~~~~~~~~~~~~')
#         print(df.isnull().sum())
#         print()
#         print('~~~~~~~~~~~~~~~~~~~~~~')
#         print("  General Statistics")
#         print('~~~~~~~~~~~~~~~~~~~~~~')
#         print(df.describe()) # basic statistics
#         print()

#     # plot pairwise correlation as heatmap
#     def plot_corr(self, title="Pairwise Correlation", cols=[]):
#         cols = self.check_cols(cols)
#         if cols == []:
#             cols = self.main_cols

#         if self.time in cols:
#             raise Exception(f"Error. No '{self.time}' collumn key allowed for pairwise correlations.")

#         corr = self.df[cols].corr()
#         plt.figure()
#         plt.title(title)
#         sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={ 'size':9})

#     # plot pairwise correlation of rolling avg
#     def plot_corr_roll(self, num):
#         if num not in self.roll_avg_nums:
#             raise Exception(f'Error! {num}{self.time} rolling average not created yet. ')
#         cols = self.get_cols_roll(num)
#         self.plot_corr(title=f"{num}{self.time} Rolling Correlation",cols=cols)

#     # Time Series Plot
#     def plot_time(self,title="Time Series",variable="variable", value="value", cols=[]):
#         cols = self.check_cols(cols)
#         if cols == []:
#             cols = [self.time, *self.main_cols]
#         if self.time not in cols:
#             raise Exception(f'Error! "{self.time}" column key is required for a time series plot.')

#         # Restructure dataframe for plot
#         df_melt = pd.melt(self.df[cols], [self.time])
#         df_melt = df_melt.rename({"value": value, "variable": variable }, axis=1)
#         plt.figure()
#         plt.title(title)
#         sns.lineplot(data=df_melt, x=self.time, y=value, hue=variable)
#         plt.show()
        
#     # time series plot for rolling average
#     def plot_time_roll(self, num, title="Time Series",variable="variable", value="value"):
#         if num not in self.roll_avg_nums:
#             raise Exception(f'Error! {num}{self.time} rolling average not created yet. ')
#         cols = self.get_cols_roll(num)
#         self.plot_time(title, variable, value, cols=[self.time, *cols])
        
