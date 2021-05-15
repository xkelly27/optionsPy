import pandas as pd


def to_mult_2(n):
    if n%2 != 0:
        return n + 1
    return n


def create_range(upper):
    return [x for x in range(0, upper, 1)]


def long_option_payoff(strike, type, cost, rng):
    r = create_range(rng)
    y_val = []
    for i in r:
        if type == "call":
            a = (max((i - strike), 0) - cost) * 100
            y_val.append(a)
        if type == "put":
            a = (max((strike - i), 0) - cost) * 100
            y_val.append(a)
    return pd.Series(y_val)


def short_option_payoff(strike, type, cost, rng):
    r = create_range(rng)
    y_val = []
    for i in r:
        if type == "call":
            a = -(max((i - strike), 0) - cost) * 100
            y_val.append(a)
        if type == "put":
            a = -(max((strike - i), 0) - cost) * 100
            y_val.append(a)
    return pd.Series(y_val)


def long_equity_payoff(cost, rng):
    r = create_range(rng)
    y_val = []
    for i in r:
        a = (i - cost) * 100
        y_val.append(a)
    return pd.Series(y_val)


def short_equity_payoff(cost, rng):
    r = create_range(rng)
    y_val = []
    for i in r:
        a = (cost - i) * 100
        y_val.append(a)
    return pd.Series(y_val)


def combine(*args):
    l = []
    for arg in args:
        l.append(arg)
    return pd.concat(l, axis=1)


def payoff(data):
    data['profit'] = data.sum(axis=1)
    return data['profit']

