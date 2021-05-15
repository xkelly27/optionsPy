import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as plticker

from matplotlib.dates import DateFormatter

sns.set()

data = pd.read_csv("/Users/xavierkelly/Downloads/smile_history.csv", skiprows=1, infer_datetime_format=True)
data["Dates"] = data.set_index("Date", inplace=True)
data.index = pd.to_datetime(data.index, infer_datetime_format=True)
data = data.drop(data.columns[[-1]], axis=1)

fig, ax = plt.subplots()
ax.plot(data)
date_form = DateFormatter("%m-%Y")
ax.xaxis.set_major_formatter(date_form)
plt.show()
