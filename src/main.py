import naive
import arima
import svr

import csv
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

DEBUG_PRINTS=1

def loadExcelData(filepath):
    print("Loading .xlsx data...")
    excelData = pd.ExcelFile(filepath)
    data = excelData.parse()
    return data


def convert_datetime(data, feature):
    data.index = data['YEAR'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d'))
    del data['YEARM']
    del data['YEAR']
    del data['MONTH']
    del data['DAY']
    print(data.head())
    return data



def difference(data, order=1):
    data_diff = data.diff(order)
    return data_diff[order:]

#def inv_difference(pre):


def check_stationary(data):
    out = adfuller(data)
    p_val = out[1]
    print("p-val:", p_val)
    if p_val < 0.05:
        return True
    else:
        return False

def data_visualization(data, feature):

    tmp = data[feature][data[feature].notnull()]

    plt.plot(tmp.index, tmp, label = feature)
    plt.legend()
    plt.title(feature + " time series plot")
    plt.xlabel("years")
    plt.ylabel(feature)
    plt.savefig("m1_plots/" + feature + "_time-series.png")
    plt.show()
    plt.clf()

    plt.boxplot(tmp)
    plt.title("CO2 box plot")
    plt.ylabel(feature)
    plt.savefig("m1_plots/"+ feature +"_boxplot.png")
    plt.show()
    plt.clf()


    # seasonal decomposition
    sd = seasonal_decompose(tmp, model= "additive")
    fig, ax = plt.subplots(4,figsize=(12,9))
    ax[0].plot(sd.observed, color='red', label = "observed")
    ax[0].legend()
    ax[1].plot(sd.seasonal, color = 'green', label = "seasonal")
    ax[1].legend()
    ax[2].plot(sd.resid, color = 'black', label = "residual")
    ax[2].legend()
    ax[3].plot(sd.trend, color = 'blue', label= "trend")
    ax[3].legend()
    fig.suptitle(feature + " seasonal decomposition")
    fig.savefig("m1_plots/" + feature + "_seasonal_decompose.png")
    plt.show()
    plt.clf()


def main():
    print('Start Program...')

    # data-loading
    #co2_data = loadExcelData("..\data\world_co2_gdp.xlsx")
    co2_data = loadExcelData("..\data\co2_monthly_process.xlsx")
    if DEBUG_PRINTS:
        print("LOADED DATA: type:", type(co2_data), "shape:", len(co2_data))


    co2_df = convert_datetime(co2_data, "CO2")

    # data pre-processing
    #co2_data_process = data_preprocess(co2_data, country)
    # pre-process (convert to date-time series)
    #co2_data.index = co2_data['YEAR'].apply(lambda x: dt.datetime(int(x),1,1,0,0,0))
    #del co2_data['YEAR']

    print(co2_df.head())

    # data visualization
    #for label in co2_data.columns:
    data_visualization(co2_df, "CO2")


    is_stationary = check_stationary(co2_df)
    print("data(orig) is_stationary:", is_stationary)


    if is_stationary == False:
        co2_df["CO2_DIFF"] = difference(co2_df, 1)


    co2_df = co2_df[co2_df["CO2_DIFF"].notnull()]
    print(co2_df.head())


    is_diff_stationary = check_stationary(co2_df["CO2_DIFF"])
    if is_diff_stationary == True:
        print("data(diff) is_stationary:", is_diff_stationary)

    # rolling average(naive) model
    if is_stationary == False:
        naive.fitRollingAvgModel(co2_df, "CO2_DIFF", rm=2)
    else:
        naive.fitRollingAvgModel(co2_df, "CO2", rm=2)

    #arima-model
    arima.train_and_predict(co2_df,"CO2_DIFF")

    #run varma-model
    #varma.train_and_predict(co2_data)

    # run svr-model
    svr.train_and_predict(co2_df, "CO2")

    # run svr-model
    #lstm.train_and_predict(co2_data)

    print('End Program...')

if __name__ == '__main__':
    main()



