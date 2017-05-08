from statsmodels.tsa.stattools import adfuller
from matplotlib import pylab
import matplotlib.pylab as plt
import pandas as pd


def test_stationarity(timeseries, k, original='Original'):

    # Determing rolling statistics
    rolmean = timeseries.rolling(window=k, center=False).mean()
    rolstd = timeseries.rolling(window=k, center=False).std()

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label=original)
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #  GET iothub.nasys.no/api/node/{AuthKey}/{offset}/{limit}/{asc/desc}

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    # import collections
    # collections.Counter(x) == collections.Counter(y)
