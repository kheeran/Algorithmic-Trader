import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import linear_model



# load data
data = pd.read_csv('/home/kheeran/Documents/PythonRefresher/HackerRankRegression/trainingdata.txt', header=None)
data.columns=["charge_time", "watch_time"]

# # visualise data
# sns.relplot(
#     data=data,
#     x="charge_time", y ="watch_time"
# )

# data is linear until max point 
# find max watch time
max = data['watch_time'].max()

# add column to highlight maxed rows and group accordingly
data['maxed'] = data.apply(lambda row: row.watch_time >= max, axis=1)
grouped = data.groupby(data.maxed)
flat_data = grouped.get_group(True)
linear_data = grouped.get_group(False)

# find change point
change_point = flat_data['charge_time'].min()



# learn linear model

X = np.array([ [x] for x in linear_data['charge_time']])
y = np.array([ [x] for x in linear_data['watch_time']])
reg = linear_model.LinearRegression(fit_intercept=False).fit(X,y)

# predict new data
reg.predict()


