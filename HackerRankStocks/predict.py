import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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
maincols = list(data_dict.keys())


### Trading Strategy: Initially split the money up equally between all stocks. For each stock, trade with an independent trading strategy using the predictions. Every N=10 trades, re-weight total money to each stock according to success.
# => we need separate predictions for each stock

# Attempt 1: Trained using prices of stock
# Attempt 2: Trained using daily price difference

# useful next day label function
def next_day_label(col):
    return f'{col} (next day)'

# add next day features
for col in maincols:
    new_name = next_day_label(col)
    maindf[new_name] = maindf[col].shift(-1) # Attempt 1

# training and evaluating models for each stock
def train_eval(customdf, traincols=maincols, test_size=0.8, plot=True):

    # drop any row with a null entry
    customdf = customdf.dropna()

    ## Double check correlation with next day features
    # i.e., predictive power
    if plot:
        corr = customdf.corr()
        plt.figure()
        sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={'size': 5})
        plt.show()


    ## Create and evaluate each predictor
    models = {}
    r2scores = {}

    # observed variables (same for all predictors)
    X = customdf[traincols]
    for col in maincols:
        # target variables (different for each predictor)
        y = customdf[[next_day_label(col)]]

        # train/test split (unshuffled for time series)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        # fit regression model (random forest) and store for later
        model = RandomForestRegressor()
        model.fit(X_train.values, y_train.values.ravel())
        models[col] = model

        # get predictions
        y_pred_vals = model.predict(X_test.values) # get predictions

        # evaluate with R2 score and store for later
        r2score = r2_score(y_test.values, y_pred_vals)
        r2scores[col] = r2score

        if plot:
            # plot test vs predicted
            plot_df = pd.DataFrame.from_dict({'ground': y_test.values.ravel(), 'prediction': y_pred_vals})

            # Sanity check for plot
            # print(f"Predictions for {col}")
            # print(plot_df.describe())
            # print(plot_df.info())
            # print("Num of nulls:")
            # print(plot_df.isnull().sum())

            # add day column
            plot_df.insert(0, column='test_day', value=[ i + 1 for i in range(y_pred_vals.shape[0])])

            # melt dataframe for plotting and plot
            plot_df = pd.melt(plot_df, ['test_day'])
            plt.figure()
            plt.title(f"{col} Model with R2 = {r2score}")
            sns.lineplot(data=plot_df, x="test_day", y="value", hue="variable")
            plt.xlabel("Test Day")
            plt.ylabel("Price")
            plt.show()

    return models, r2scores


# Attempt 1: Using price data directly
print("ATTEMPT 1")
print(maindf.head())
models1, r2scores1 = train_eval(maindf, plot=True)

# Attempt 2: Using price differences
print("ATTEMPT 2")

diffdf = maindf.diff()
print(diffdf.head())
models2, r2scores2 = train_eval(diffdf, plot=True)

# Attempt3: Using price differences only for targets
print("ATTEMPT 3")

partdiffdf = maindf
for col in maincols:
    partdiffdf[next_day_label(col)] = partdiffdf[next_day_label(col)].diff()

print(partdiffdf.head())
models3, r2scores3 = train_eval(partdiffdf, plot=True)

# Attempt4: Using both prices and price differences for training and price differences for targets
print("ATTEMPT 4")

alldf = maindf

def diff_label(col):
    return f'{col} diff'

for col in maincols: 
    alldf[diff_label(col)] = alldf[col].diff()

traincols4 = [*maincols, *[diff_label(col) for col in maincols]]

print(alldf.head())
models4, r2scores4 = train_eval(alldf, traincols=traincols4, plot=True)

# # Create dataframe to compare r2scores of different attempts

stock = []
attempt1 = []
attempt2 = []
attempt3 = []
attempt4 = []

for col in maincols:
    stock.append(col)
    attempt1.append(r2scores1[col])
    attempt2.append(r2scores2[col])
    attempt3.append(r2scores3[col])
    attempt4.append(r2scores4[col])

# create dataframe:
r2_df = pd.DataFrame.from_dict(
    data={
        'stock' : stock,
        'attempt1' : attempt1,
        'attempt2' : attempt2,
        'attempt3' : attempt3,
        'attempt4' : attempt4
    },
)

# print table
print(r2_df)

# plot bar chart
r2_df_melt = pd.melt(r2_df, ['stock'])
plt.figure()
plt.title("R2 Scores for Attempts")
sns.barplot(data=r2_df_melt, x='stock', y='value', hue='variable')
plt.ylim(-2,1)
plt.show()
