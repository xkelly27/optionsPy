import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si


class option_greeks:
    def __init__(self, volatility, strike, underlying, interest_rate=0.01, time=0.125):
        self.volatility = volatility
        self.strike = strike
        self.underlying = underlying
        self.interest_rate = interest_rate
        self.time = time

    def d1(self):
        return np.log(self.underlying / self.strike) + ((self.interest_rate + (self.volatility**2) / 2) * self.time)

    def continuous_d1(self):
        return np.exp(-(self.d1()**2) / 2) / np.sqrt(2 * np.pi)

    def continuous_d2(self):
        return self.continuous_d1() - self.volatility * np.sqrt(self.time)

    def delta_call(self):
        return np.exp()

    def vega(self):
        return (1 / 100) * self.strike * self.continuous_d1() * np.sqrt(self.time)

    def rho(self):
        return (1 / 100) * self.strike * self.time * np.exp(-self.interest_rate * self.time) * self.continuous_d2()


def standard_norm(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-(x**2) / 2)



c = 9
strike = 100
i = 0.04
t = 0.5
d = 1
p = 5
spot = 103

#
# print(c + (strike * np.exp(-0.04*0.5)))
# print(p + (spot) - d)



a = (0.24 /16)
m = (a * 80)**2
g = -1000
# print(0.5 * g * m)

def theta(gamma, spot, iv, trading_days=256):
    return 0.5 * gamma * ((spot * iv / np.sqrt(trading_days))**2)

def gamma(gamma, move):
    return 0.5 * gamma * (move**2)


hsbc_theta = theta(200, 70, 0.32)
hsbc_gamma = gamma(6, 1.5)
print(hsbc_theta)