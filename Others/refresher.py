from scipy.stats import rv_continuous
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

class gaussian_gen(rv_continuous):
    # Normal Distribution
    def _pdf(self,x):
        return np.exp(-x**2/2) / np.sqrt(2 * np.pi)

# initialise gaussian noise dist
noise = gaussian_gen()

# generate linear data with some noise
m = 2.6
xs = np.array([i + 1 for i in range(0,100,4)])
ys = np.array([m*x + noise.rvs(scale=5) for x in xs])

# plot data
plt.figure()
plt.plot(xs, ys, label="observed")

# fit linear regression
reg = LinearRegression(fit_intercept=False).fit(xs.reshape(-1,1), ys)
plt.plot(xs, reg.predict(xs.reshape(-1,1)), label="regression")
plt.show()

print(reg.coef_, reg.score(xs.reshape(-1,1), ys))
