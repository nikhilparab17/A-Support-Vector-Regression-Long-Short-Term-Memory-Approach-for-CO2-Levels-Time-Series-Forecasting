import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

print("Importing arima forecasting model...")

DEBUG_MODEL = 1

# calculate 'p' using Auto Correlation Function(ACF)
def acf(x, feature = 'CO2', l=3):
    plot_acf(x[feature], lags=l)
    plt.title("Auto Correlation Function plot [ CO2, lag= " + str(l) + "]")
    plt.savefig("m1_plots/CO2_ACF.png")
    plt.show()
    plt.clf()


# calculate 'q' using Partial Auto Correlation Function(PACF)
def pacf(x, feature ='CO2', l=3):
    plot_pacf(x[feature], lags=l)
    plt.title("Partial Auto Correlation Function plot [ CO2, lag= " + str(l) + "]")
    plt.savefig("m1_plots/CO2_PACF.png")
    plt.show()
    plt.clf()


def check_stationary(data):
    out = adfuller(data)
    p_val = out[1]
    print("p-val:", p_val)
    if p_val < 0.05:
        return True
    else:
        return False

def difference(x,feature, interval):
    diff = x
    #print(diff.head())
    temp = diff.shift(interval)
    #diff = diff[interval:,:]
    #print(temp.head())
    out = diff - temp
    #print(out.head())
    out = out.iloc[interval:]
    #print(out.head())
    return out

def inverse_difference(last_obs, pred_diff):
    return last_obs + pred_diff

def rms_error(true_values, predict_values):
    rms_error = np.sqrt((1.0/len(true_values))*np.dot(true_values-predict_values, true_values - predict_values))
    return rms_error

def train_and_predict(X, feature = 'CO2', seasonal = True):
    print("Running ARIMA model...\n")


    is_stationary = check_stationary(X[feature])
    if is_stationary == True:
        print("data(diff) is_stationary:", is_stationary)

    #split data into train(80%),test(20%)
    split = len(X[feature]) - int(0.2*len(X[feature]))
    print("len(X_new):", len(X[feature]), "split:", split)
    x_train = X[0:split]
    x_test = X[split:]
    print(len(x_train), len(x_test))

    p = acf(x_train, feature, 5)
    q = pacf(x_test,feature, 5)

    #training
    if not seasonal:
        model = ARIMA(x_train[feature].values, order=(3,0,2))
    else:
        model = SARIMAX(x_train[feature].values, order=(2, 1, 2), seasonal_order=(0, 1, 2, 4))

    model_fit = model.fit(disp=False)

    #diff-prediction
    pred = model_fit.predict(len(x_test[feature]))
    pred_diff = pd.DataFrame(x_test)
    pred_diff['predictions'] = pred[0:len(x_test[feature])]

    xorig_test = X["CO2"][split-1]

    #orig-prediction
    pred_diff['st_predictions'] = pred_diff[feature][0:len(x_test[feature])]
    pred_diff['lt_predictions'] = pred_diff[feature][0:len(x_test[feature])]
    pred_diff['true_values'] = X[feature][split:]
    s = 0
    for i in range(len(x_test)):
        s = s + pred_diff['predictions'].iloc[i]
        pred_diff['lt_predictions'].iloc[i] = s + X["CO2"][split]
        pred_diff['st_predictions'].iloc[i] = pred_diff['predictions'].iloc[i] + X["CO2"][split + i]


    # plot diff-prediction
    plt.clf()
    plt.plot(X["CO2"])
    #plt.plot(pred_diff['predictions'])
    plt.plot(pred_diff['lt_predictions'], label='long-term forecast')
    plt.plot(pred_diff['st_predictions'], label='short-term forecast')
    plt.xlabel('years')
    plt.ylabel('co2 (ppm)')
    plt.title("CO2 Forecast [ARIMA Model]")
    plt.legend()
    plt.savefig("m1_plots/co2_arima_forecast_model.png")
    plt.show()
    plt.clf()


    # plot st-prediction
    #plt.clf()
    #plt.plot(X[country])
    #plt.plot(pred_diff['st_predictions'], label = 'short-term forecast')
    #plt.legend()
    #plt.plot(pred_diff['lt_predictions'], label = 'long-term forecast')
    #plt.title(country + " [arima forecast model]")
    #plt.legend()
    #plt.savefig(country + "_arima-model.png")
    #plt.show()
    #plt.clf()

    #plt.clf()


    # plot st-prediction
    #plt.clf()
    #plt.plot(X[country])
    #plt.plot(pred_diff['lt_predictions'])
    #plt.show()
    #plt.savefig(country + "lt_ARIMA.png")
    #plt.clf()

    # calculate prediction rms error
    pred_error = rms_error(X['CO2'][split:], pred_diff['st_predictions'])
    print("rms(st):" , pred_error)

    # calculate prediction rms error
    pred_error = rms_error(X['CO2'][split:], pred_diff['lt_predictions'])
    print("rms(lt):" , pred_error)
