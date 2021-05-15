import pandas as pd
import numpy as np


class option_greeks:
    def __init__(self, volatility, strike, underlying, interest_rate=0.01, time=0.125):
        self.volatility = volatility
        self.strike = strike
        self.underlying = underlying
        self.interest_rate = interest_rate
        self.time = time

    def d1(self):
        return np.log(self.underlying / self.strike) + ((self.interest_rate + (self.volatility**2) / 2) * self.time)

    def continuous(self):
        return np.exp(-(self.d1()**2) / 2) / np.sqrt(2 * np.pi)

    def vega(self):
        return self.strike * self.continuous() * np.sqrt(self.time)



SPX_call = option_greeks(0.2, 3283.66*1.05, 3283.66, 0.01, 0.125).vega()
SPX_put = option_greeks(0.2, 3283.66*0.95, 3283.66, 0.01, 0.125).vega()


SPX_prices = [3248, 3090, 3232]


HSCEI_prices = [10268, 10485, 10008]
HSCEI_converter = 0.1286

HSCEI_vega = []


def putcall_vega(price):
    dicts = {}
    dicts["Call"] = option_greeks(0.2, price*1.05, price, 0.01, 0.125).vega()
    dicts["Put"] = option_greeks(0.1, price * 0.95, price, 0.01, 0.125).vega()
    return dicts


def convert(data, factor):
    X = []
    for i in HSCEI_vega:
        X.append(i * HSCEI_converter)
    return X


SPX_vega = putcall_vega(SPX_prices[0])
# print(SPX_vega)

c = 25000*2152.65
d = c * 0.0008
print(d)