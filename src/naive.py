import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA

print("Importing rolling-avg(naive) forecasting model...")

DEBUG_MODEL= 0


def rms_error(true_values, predict_values):
    rms_error = np.sqrt((1.0/len(true_values))*np.dot(true_values-predict_values, true_values - predict_values))
    return rms_error

def inv_diff(prev_val, data):
    data_orig = data.copy()
    data_orig[0] = prev_val + data[0]
    for i in np.arange(1,len(data)):
        data_orig[i] = data_orig[i] + data_orig[i-1]
    return  data_orig


def fitRollingAvgModel(X, feature='CO2', rm = 2):
    print("Running rolling avg model...")

    hist_value = X["CO2"][rm-1]
    true_values = X[feature][rm:]
    predict_values = X[feature].rolling(rm).mean().shift(1)[rm:]



    split = len(X[feature]) - int(0.2*len(X[feature]))
    x_train = X[feature][:split]
    x_test = X[feature][split:]
    model = ARMA(x_train,order=(0,rm))
    model_fit = model.fit(disp=False)
    pred_test = model_fit.predict(split, split + len(x_test) -1)

    if DEBUG_MODEL:
        print(orig_values.head(), len(orig_values))
        print(true_values.head() , len(true_values))
        print(predict_values.head(), len(predict_values))

    true_values_orig = inv_diff(hist_value, true_values)
    predict_values_orig = inv_diff(hist_value,predict_values)

    pred_test_hist = X["CO2"][split-1]
    pred_test_orig = inv_diff(pred_test_hist, pred_test)

    print(true_values.head())
    print(true_values_orig.head())
    print(predict_values_orig.head())
    print(pred_test_orig)


    plt.plot(X["CO2"])
    plt.plot(predict_values_orig[-len(pred_test):], label='short-term forecast' )
    plt.plot(pred_test_orig, label = 'long-term forecast')
    plt.xlabel('years')
    plt.ylabel('co2 (ppm)')
    plt.title("CO2 Rolling Average Time Series Forecast Model")
    plt.legend()
    plt.savefig("m1_plots/co2_rolling_avg_forecast_model.png")
    plt.show()
    plt.clf()

    st_error = rms_error(predict_values_orig[-len(pred_test):], X["CO2"][-len(pred_test):])
    lt_error = rms_error(pred_test_orig, X["CO2"][-len(pred_test):])

    print("forecasting-model: naive(rolling-avg)", "lag:", rm, "rms-error(st):", st_error)
    print("forecasting-model: naive(rolling-avg)", "lag:", rm, "rms-error(lt):", lt_error)



    plt.plot(true_values)
    plt.plot(predict_values[-len(pred_test):], label='short-term forecast' )
    plt.plot(pred_test, label = 'long-term forecast')
    plt.xlabel('years')
    plt.ylabel('co2 (ppm)')
    plt.title("CO2 Rolling Average (diff) Time Series Forecast Model")
    plt.legend()
    plt.savefig("m1_plots/co2_emission_rolling_avg_diff_forecast_model.png")
    plt.show()
    plt.clf()

    st_diff_error = rms_error(true_values[split:] , predict_values[split:])
    lt_diff_error = rms_error(x_test, pred_test)
    #print("forecasting-model: naive(rolling-avg)", "lag:", rm, "rms-error(st):", st_diff_error)
    #print("forecasting-model: naive(rolling-avg)", "lag:", rm, "rms-error(lt):", lt_diff_error)

if __name__ == '__main__':
    model1()
