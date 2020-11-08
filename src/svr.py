import pandas as pd
import numpy as np

import datetime as dt
import matplotlib.dates as md
import matplotlib.pyplot as plt
from sklearn.svm import SVR

from sklearn.preprocessing import MinMaxScaler

print("Importing support-vector-regression(svr) forecasting model...")

DEBUG_MODEL= 0


def rms_error(true_values, predict_values):
    rms_error = np.sqrt((1.0/len(true_values))*np.dot(true_values-predict_values, true_values - predict_values))
    return rms_error


def train_and_predict(df, feature='CO2'):

    df["date_int"] = df.index.map(md.date2num)
    print(df.head())

    dates_np = df["date_int"].to_numpy()
    co2_np = df[feature].to_numpy()

    # Convert to 1d Vector
    dates = np.reshape(dates_np, (len(dates_np), 1))
    co2 = np.reshape(co2_np, (len(co2_np), 1))


    # sacling data
    scaler_in = MinMaxScaler()  # for inputs
    scaler_out = MinMaxScaler()  # for outputs

    x_scale = scaler_in.fit_transform(dates[:, 0].reshape(-1, 1))
    y_scale = scaler_out.fit_transform(co2[:, 0].reshape(-1, 1))

    slice = len(dates_np) - int(0.2*len(dates_np))
    print(slice)
    x_train, y_train = x_scale[:slice,:],y_scale[:slice,:]
    x_test, y_test = x_scale[slice:], y_scale[slice:]



    print(np.shape(x_train), np.shape(y_train))
    print(x_train[0:10,:], y_train[:10,:])

    #svr_rbf = SVR(kernel='rbf', C=5, gamma = 0.01)

    svr_lin  = SVR(kernel='linear', epsilon=0.001, C=1000, gamma= 0.1)
    svr_poly = SVR(kernel='poly', epsilon=0.001, C= 1000, gamma=0.1)
    svr_rbf = SVR(kernel='rbf', epsilon=0.001, C=1000, gamma=0.1)

    #print()
    t = 1

    svr_rbf.fit(x_train,y_train)
    predict_rbf = svr_rbf.predict(x_test)

    svr_poly.fit(x_train,y_train)
    predict_poly = svr_poly.predict(x_test)


    svr_lin.fit(x_train,y_train)
    predict_lin = svr_lin.predict(x_test)

    predict_rbf_rescale = scaler_out.inverse_transform(predict_rbf.reshape(-1,1))
    predict_poly_rescale = scaler_out.inverse_transform(predict_poly.reshape(-1,1))
    predict_lin_rescale = scaler_out.inverse_transform(predict_lin.reshape(-1,1))

    plt.figure(figsize=(12,6))
    plt.plot(df.index[:slice], co2[:slice], label='data')
    plt.plot(df.index[slice:], co2[slice:], label='expected')
    plt.plot(df.index[slice:], predict_rbf_rescale, label='forecast (kernel: rbf)')
    plt.plot(df.index[slice:], predict_poly_rescale, label='forecast (kernel:polynomial)')
    plt.plot(df.index[slice:], predict_lin_rescale, label='forecast (kernel:linear)')
    plt.xlabel('years')
    plt.ylabel('co2 (ppm)')
    plt.title("CO2 Forecast [Support Vector Regression Model]")
    plt.legend()
    plt.savefig("m1_plots/co2_svr_forecast_model.png")
    plt.show()
    plt.clf()

    a = co2[slice:]
    b = predict_lin_rescale
    print(np.shape(a), np.shape(b))
    error_rbf = rms_error(co2[slice:,0], predict_rbf_rescale[:,0])
    error_poly = rms_error(co2[slice:,0], predict_poly_rescale[:,0])
    error_lin = rms_error(co2[slice:,0], predict_lin_rescale[:,0])


    print("forecasting-model: SVR (kernel:rbf)  rms-error:", error_rbf)
    print("forecasting-model: SVR (kernel:poly)  rms-error:", error_poly)
    print("forecasting-model: SVR (kernel:linear)  rms-error:", error_lin)



if __name__ == '__main__':
    model1()
