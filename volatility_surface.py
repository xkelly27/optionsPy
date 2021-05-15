
# BSM option valuation
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


## Helper functions ##
def dN(x):
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def N(d):
    return quad(lambda x: dN(x), -20, d, limit=50)[0]


def d1f(St, K, t, T, r, sigma):

    d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2)
          * (T - t)) / (sigma * math.sqrt(T - t))
    return d1


def BSM_call_value(St, K, t, T, r, sigma):
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    call_value = St * N(d1) - math.exp(-r * (T - t)) * K * N(d2)
    return call_value


def BSM_put_value(St, K, t, T, r, sigma):
    put_value = BSM_call_value(St, K, t, T, r, sigma) - St + math.exp(-r * (T - t)) * K
    return put_value


# Test Option Payoff and Time value
K = 8000  # Strike price
T = 1.0  # time-to-maturity
r = 0.025  # constant, risk-free rate
vol = 0.2  # constant volatility

# Generate spot prices
S = np.linspace(4000, 12000, 150)
h = np.maximum(S - K, 0)  # payoff of the option
C = [BSM_call_value(Szero, K, 0, T, r, vol) for Szero in S]  # BS call option values

plt.figure()
plt.plot(S, h, 'b-.', lw=2.5, label='payoff')  # plot inner value at maturity
plt.plot(S, C, 'r', lw=2.5, label='present value')  # plot option present value
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('index level S0')
plt.ylabel('present value C(t=0)')
plt.show()

import pandas as pd

# model parameters
S0 = 100.0  # initial index level
T = 10.0  # time horizon
r = 0.05  # risk-less short rate
vol = 0.2  # instantaneous volatility
# simulation parameters
np.random.seed(250000)
# generate a pd array with business dates, ignores holidays
gbm_dates = pd.date_range(start='10-05-2007', end='10-05-2017', freq='B')
M = len(gbm_dates)  # time steps
I = 1  # index level paths
dt = 1 / 252.  # 252 business days a year
df = math.exp(-r * dt)  # discount factor

# stock price paths
rand = np.random.standard_normal((M, I))  # random numbers
S = np.zeros_like(rand)  # stock matrix
S[0] = S0  # initial values
for t in range(1, M):  # stock price paths using Eq.5
    S[t] = S[t - 1] * np.exp((r - vol ** 2 / 2) * dt + vol * rand[t] * math.sqrt(dt))
# create a pd dataframe with date as index and a column named "spot"
gbm = pd.DataFrame(S[:, 0], index=gbm_dates, columns=['spot'])
gbm['returns'] = np.log(gbm['spot'] / gbm['spot'].shift(1))  # log returns
# Realized Volatility
gbm['realized_var'] = 252 * np.cumsum(gbm['returns'] ** 2) / np.arange(len(gbm))
gbm['realized_vol'] = np.sqrt(gbm['realized_var'])
gbm = gbm.dropna()

plt.figure(figsize=(9, 6))
plt.subplot(211)
gbm['spot'].plot()
plt.ylabel('daily quotes')
plt.grid(True)
plt.axis('tight')
plt.subplot(212)
gbm['returns'].plot()
plt.ylabel('daily log returns')
plt.grid(True)
plt.axis('tight')
plt.show()


headers = ['Date', 'Strike', 'Call_Bid', 'Call_Ask', 'Call', 'Maturity', 'Put_Bid', 'Put_Ask', 'Put']
dtypes = {'Date': 'str', 'Strike': 'float', 'Call_Bid': 'float', 'Call_Ask': 'float',
          'Call': 'float', 'Maturity': 'str', 'Put_Bid': 'float', 'Put_Ask': 'float', 'Put': 'float'}
parse_dates = ['Date', 'Maturity']

data = pd.read_csv('/Users/xavierkelly/Downloads/AAPL_BBG_vols.csv', skiprows=1, header=None, names=headers, dtype=dtypes, parse_dates=parse_dates)
data['BS_Imp_Vol'] = 0.0
data['TTM'] = 0.0
r = 0.0248  # risk-free rate
S0 = 153.97  # spot price as of 5/11/2017
div = 0.0182
for i in data.index:
    t = data['Date'][i]
    T = data['Maturity'][i]
    K = data['Strike'][i]
    Put = data['Put'][i]
    time_to_maturity = (T - t).days / 365.0
    data.loc[i, 'TTM'] = time_to_maturity

    def func_BS(sigma):
        return BSM_put_value(S0, K, 0, time_to_maturity, r, sigma) - Put

    bs_imp_vol = brentq(func_BS, 0.03, 1.0)
    data.loc[i, 'BS_Imp_Vol'] = bs_imp_vol


ttm = data['TTM'].tolist()
strikes = data['Strike'].tolist()
imp_vol = data['BS_Imp_Vol'].tolist()
f = interp2d(strikes, ttm, imp_vol, kind='cubic')

plot_strikes = np.linspace(data['Strike'].min(), data['Strike'].max(), 25)
plot_ttm = np.linspace(0, data['TTM'].max(), 25)
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(plot_strikes, plot_ttm)
Z = np.array([f(x, y) for xr, yr in zip(X, Y) for x, y in zip(xr, yr)]).reshape(len(X), len(X[0]))
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1)
ax.set_xlabel('Strikes')
ax.set_ylabel('Time-to-Maturity')
ax.set_zlabel('Implied Volatility')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()