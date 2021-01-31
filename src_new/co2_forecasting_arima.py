#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/nikhilparab17/Time-Series-Forecasting-for-CO2-Levels/blob/master/src/co2_forecasting_arima.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


#get_ipython().system(' pip install pmdarima')

#!pip install pmdarima

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.stattools import adfuller


# Dataset
DATASET = "../data/co2_dataset_preprocess.xlsx"

# function to load excel data
def loadExcelData(filepath):
    print("Loading .xlsx data...")
    excelData = pd.ExcelFile(filepath)
    data = excelData.parse()
    return data

# function to convert to date-time series
def convert_datetime(data):
    data.index = data['YEAR'].apply(lambda x: dt.datetime.strptime(x, '%Y/%m/%d'))
    del data['YEAR']
    return data

# difference operation
def difference(data,order=1):
    data_diff = data.diff(order)
    return data_diff[order:]

# stationarity test
def check_stationary(data):
    out = adfuller(data)
    p_val = out[1]
    print("p-val:", p_val)
    if p_val < 0.05:
        return True
    else:
        return False

# data visualization
def data_visualization(data, feature):

    # time series
    plt.plot(data[feature].index, data[feature], label = feature)
    plt.legend()
    plt.xlabel("Years")
    plt.ylabel(feature + " (ppm)")
    plt.savefig(feature + "_time-series.png")
    plt.show()
    plt.close()

    # box plot
    plt.boxplot(data[feature])
    plt.ylabel(feature + " (ppm)")
    plt.savefig(feature +"_boxplot.png")
    plt.show()
    plt.close()

    # seasonal decomposition
    sd = seasonal_decompose(data[feature], model= "additive")
    fig, ax = plt.subplots(4,figsize=(12,9))
    ax[0].plot(sd.observed, color='red', label = "Observed")
    ax[0].legend()
    ax[1].plot(sd.seasonal, color = 'green', label = "Seasonal")
    ax[1].legend()
    ax[2].plot(sd.resid, color = 'black', label = "Residual")
    ax[2].legend()
    ax[3].plot(sd.trend, color = 'blue', label= "Trend")
    ax[3].legend()
    fig.savefig(feature + "_seasonal_decompose.png")
    plt.show()
    plt.close()

# calculate 'p' using Auto Correlation Function(ACF)
def acf(x, feature = 'CO2 Levels', l=3):
    plot_acf(x[feature], lags=l)
    plt.xlabel("Lag")
    plt.savefig("CO2_ACF.png")
    plt.close()

# calculate 'q' using Partial Auto Correlation Function(PACF)
def pacf(x, feature ='CO2 Levels', l=3):
    plot_pacf(x[feature], lags=l)
    plt.xlabel("Lag")
    plt.savefig("CO2_PACF.png")
    plt.close()

# train and fit ARIMA model
def train_and_fit_arima(x, test_split = 0.2):

    # run auto-arima grid search 
    stepwise_model= auto_arima(x, exogenous=None, start_p=0, d=1, start_q=0,
                               max_p=3, max_d=1, max_q=3,
                               start_P=0, D=1, start_Q=0, max_P=3, max_D=3, 
                               max_Q=3, max_order=10, m=12, seasonal=True,
                               trace=True,error_action='ignore',
                               suppress_warnings=True,stepwise=False,
                               approximation=False)

    print(stepwise_model.aic())
    print(stepwise_model.summary())

    split=len(x) - int(test_split * len(x))
    train = x[0:split]
    test = x[split:]

    stepwise_model.fit(train)

    future_forecast = stepwise_model.predict(n_periods=len(test))
    future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])
    lineObjects=plt.plot(pd.concat([test, future_forecast], axis=1))
    plt.xlabel("Years")
    plt.ylabel("CO2 Levels (ppm)")
    plt.legend(iter(lineObjects), ('CO2 Levels', 'Predictions'))
    plt.savefig("Forecast.png")
    plt.show()
    plt.close()

    line1bjects=plt.plot(pd.concat([x, future_forecast], axis=1))
    plt.xlabel("Years")
    plt.ylabel("CO2 Levels (ppm)")
    plt.legend(iter(line1bjects), ('CO2 Levels', 'Predictions'))

    plt.savefig("Forecast_conc.png")
    plt.show()
    plt.close()

    pred_error = rmse(test, future_forecast)
    print("rmse:", pred_error)

    stepwise_model.plot_diagnostics(figsize=(15, 12))
    plt.savefig("Diagnostic.png")
    plt.show()
    plt.close()

# main function
def main():
    print('Start Program...')

    # data-loading
    co2_data = loadExcelData(DATASET)

    # pre-process (convert to date-time series)
    co2_df = convert_datetime(co2_data)
    print(co2_df.head())

    # data visualization
    data_visualization(co2_df, "CO2 Levels")
    acf(co2_df, l=30)
    pacf(co2_df, l=30)

    # check stationarity
    is_stationary = check_stationary(co2_df["CO2 Levels"].values)
    if is_stationary == False:
      print("CO2 Levels is_not stationary")
      co2_dif = difference(co2_df, 1)
      is_diff_stationary = check_stationary(co2_dif["CO2 Levels"].values)
      if is_diff_stationary == True:
        print("CO2 Levels (diff) is_stationary")
    else:
      print("CO2 Levels is_stationary:")

    # train and fit using arima model for [train,test] = [0.8,0.2] split
    train_and_fit_arima(co2_df["CO2 Levels"], test_split = 0.2)

    print('End Program...')

if __name__ == '__main__':
    main()



