import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import cleaned data
# It contains the fields ['Open', 'High', 'Low', 'Close', 'Volume'] for 10 (out of 503) stocks (names saved separately) for 757 days between 2019-01-01 and 2021-12-31.
samp_size = 10
main_df = pd.read_csv(f"cleaned_{samp_size}.csv")
main_info = ['Open', 'High', 'Low', 'Close', 'Volume']
tickers = np.genfromtxt(f"cleaned_{samp_size}_tickers.csv", delimiter=',', dtype=str)

# # time series plot of data
# melt_df = pd.melt(main_df, 'Date')
# plt.figure()
# sns.lineplot(data=melt_df, x='Date', y='value', hue='variable', legend=False)
# plt.yscale('log')
# plt.ylabel('price/volume')
# plt.show()

# specify observed and target labels
obs_cols = list(main_df.columns)
target_label = lambda label, num: f"{label}_days+{num}"
targ_cols = []

# create target values for each day
for col in obs_cols:
    new_label = target_label(col,1)
    targ_cols.append(new_label)
    main_df[new_label] = main_df[col].shift(-1)

# create target values for each day
for col in obs_cols:
    new_label = target_label(col,2)
    targ_cols.append(new_label)
    main_df[new_label] = main_df[col].shift(-2)

# drop final row, i.e., row with NaN target values
main_df = main_df.dropna()

# compute correlation
corr = main_df.drop('Date', axis=1).corr()
plt.figure()
sns.heatmap(data=corr,annot=False, fmt=".2f", annot_kws={'size' : 9})
plt.show()

# heatmap shows (as expected) that the open, close, high, and low prices of the same stock are highly pos correlated. On the other hand, a stocks volume is not as well correlated with its price (interestingly). 




