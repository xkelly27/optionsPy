import matplotlib.pyplot as plt
import math
from scipy.stats import norm


def dnorm(x):
    return norm.cdf(x)


def option_price(type, sigma, ttm, spot, strike, rf, price=0):
    d1 = (1 / sigma * (math.sqrt(ttm))) * ((math.exp(spot / strike)) + (rf + (sigma ** 2) / 2) * (ttm))
    d2 = d1 - (sigma * math.sqrt(ttm))
    if type is "call":
        price = (spot * dnorm(d1)) - (strike * math.exp(-rf * ttm) * dnorm(d2))
    if type is "put":
        price = (dnorm(-d2) * strike * math.exp(-rf * ttm)) - (dnorm(-d1) * spot)
    return round(max(price, 0), 4)




thales = option_price("call", 1, 12, 100, 200, 0)
print(thales)

def plot_var(type, var, rng=100):
    y = []
    x = []
    for i in range(1, rng):
        if var == "sigma":
            y.append(option_price(type, i / 10, 1, 100, 100, 0.01))
        if var == "ttm":
            y.append(option_price(type, 0.1, i / 10, 100, 100, 0.01))
        if var == "spot":
            y.append(option_price(type, 0.1, 1, i*100, 100, 0.01))
        if var == "strike":
            y.append(option_price(type, 0.1, 1, 100, i*100, 0.01))
        if var == "rf":
            y.append(option_price(type, 0.1, 1, 100, 100, i / 100))
        x.append(i)
    plt.plot(x, y)
    plt.show()

