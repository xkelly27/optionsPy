import pandas as pd
import numpy as np
import os
import quandl

#account created august 2020
#waste of fucking time 960 bucks a year may as well track R and IV myself
token = "bZXH-zjRfWqn8DCay3gq"

def vol_fetch(ticker):
    data = quandl.get("VOL/{}".format(ticker), authtoken=token)
    return data

data = vol_fetch("SPY")


