from optionsPy import combine as C
from optionsPy import plot

# #instantiate options from combine functions
# long_call = C.long_option_payoff(100, "call",10, 300)
# long_put = C.long_option_payoff(55, "put", 5, 300)
# short_call = C.short_option_payoff(65, "call", 13)
# short_put = C.short_option_payoff(65, "put", 3)

#instantiate equities from combine functions
long_e = C.long_equity_payoff(50, )
short_e = C.short_equity_payoff(50)

#combine dataframes function
c = C.combine(short_e, long_e)
master = C.payoff(c)

#plot (from custom)
plot.custom_plot(master)
