from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from urllib.request import urlopen
from nas import naslib
from utils import io
from matplotlib.pylab import rcParams
from utils import util
import numpy as np
from numpy import random, argsort, sqrt

import pandas as pd
import matplotlib.pylab as plt
import json
import sys
import os.path
from sklearn.neighbors import NearestNeighbors

rcParams['figure.figsize'] = 15, 6
count = 576
# average of ‘k’ consecutive values depending on the frequency of time series
k = 3000  # week


def plot_data():
    if not os.path.exists(file_name):
        init_data()

    ts = get_time_series()

    print_ts(ts)
    print_usage(ts)

    print_trend_eliminated_series(ts)
    moving_avg_difference(ts)
    print_expwighted_avg(ts)
    print_ewm_diff(ts)
    print_trend_seasonality_residuals(ts)
    print_ts_log_decompose(ts)

    print_future(ts)


def print_ts_log_decompose(ts):
    ts_log = np.log(ts)
    decomposition = seasonal_decompose(ts_log, freq=k)
    residual = decomposition.resid
    ts_log_decompose = residual
    ts_log_decompose.dropna(inplace=True)
    print_ts(ts_log_decompose, original='ts_log_decompose')


def print_trend_seasonality_residuals(ts):
    ts_log = np.log(ts)
    decomposition = seasonal_decompose(ts_log, freq=k)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def print_ewm_diff(ts):
    ts_log = np.log(ts)
    expwighted_avg = ts_log.ewm(halflife=k, min_periods=0, adjust=True, ignore_na=False).mean()
    ts_log_ewma_diff = ts_log - expwighted_avg
    util.test_stationarity(ts_log_ewma_diff, k)
    plt.show()


def print_expwighted_avg(ts):
    ts_log = np.log(ts)
    expwighted_avg = ts_log.ewm(halflife=k, min_periods=0, adjust=True, ignore_na=False).mean()
    plt.plot(ts_log, label='log(ts)')
    plt.plot(expwighted_avg, label='expwighted_avg', color='red')
    plt.title('expwighted_avg')
    plt.legend(loc='best')
    plt.show()


def print_usage(ts):
    ts_diff = ts - ts.shift()
    plt.plot(ts_diff, label='Diff')
    plt.legend(loc='best')
    plt.title("Usage")
    plt.show()


def print_ts(ts, original=None):
    util.test_stationarity(ts, k=k, original=original)
    plt.show()


def moving_avg_difference(ts):
    ts_log = np.log(ts)
    moving_avg = ts_log.rolling(window=k, center=False).mean()
    ts_log_moving_avg_diff = ts_log - moving_avg
    # print(ts_log_moving_avg_diff.head(k))
    ts_log_moving_avg_diff.dropna(inplace=True)
    util.test_stationarity(ts_log_moving_avg_diff, k, original='Moving avg difference')
    plt.show()
    return ts_log


def get_time_series():
    data_df = pd.read_json(file_name, orient='records')
    complete_data = pd.DataFrame.from_records(data_df, index='date', columns=['data', 'date'])
    ts = complete_data['data']
    return ts, ts - ts.shift(), complete_data


def print_trend_eliminated_series(ts):
    plt.title("Trend eliminated series")
    ts_log = np.log(ts)
    moving_avg = ts_log.rolling(window=k, center=False).mean()
    plt.plot(ts_log, label='Trend eliminated series log(ts)')
    plt.plot(moving_avg, color='red', label='Moving avg')
    plt.legend(loc='best')
    plt.show()


def print_future(ts):
    ts_log = np.log(ts)
    ts_log_diff = ts_log - ts_log.shift()

    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Autocorrelation Function')

    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()


def get_nas_url(devEUI, authKey, response_count=sys.maxsize):
	if devEUI is None or authKey is None:
		raise("get_nas_url access parameters are not fulfilled")
    nas_url = "https://iothub.nasys.no/api/node/" + authKey + "/" + devEUI + "/RX/0/" + str(
        response_count) + "/asc"
    return nas_url


def knn_search(x, D, K):
    """ find K nearest neighbours of data among D """
    ndata = D.shape[1]
    K = K if K < ndata else ndata
    # euclidean distances from the other points
    sqd = sqrt(((D - x[:, :ndata]) ** 2).sum(axis=0))
    idx = argsort(sqd)  # sorting
    # return the indexes of K nearest neighbours
    return idx[:K]


def init_data(water_meter_id=None, file_name='usage_data.json', div=100):
    if os.path.exists(file_name):
        os.remove(file_name)

    jsonurl = urlopen(get_nas_url(water_meter_id))

    response = json.loads(jsonurl.read())
    total = response['total']

    usage_data_list = response['result']
    usage_data_list = list(map(naslib.map_nas_data(div), filter(lambda row: row['fport'] == '24', usage_data_list)))

    io.write_to_file(file_name, usage_data_list)
