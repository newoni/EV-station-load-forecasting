import pandas as pd
import datetime as dt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lib.preprocessing import Preprocessing4TimeSeriesAnalysis
from lib.time_series_analysis import TimeSeriesAnalysis
from lib.graphs import Graphs
from lib.SVR import SVRmodel

if __name__ =="__main__":
    '''
    Preprocessing Data
    '''

    # Data load
    preprocessing = Preprocessing4TimeSeriesAnalysis()

    # Data preprocessing
    #preprocessing.iter_oper()      # preprocess the data

    # Load preprocessed data
    preprocessing.load_preprocessed_data()
    data = preprocessing.resampled_data_1hour

    # Select the horizon of data
    data = preprocessing.cutDataOnTime(data, '2017-01-01', '2017-09-30')
    # plt.figure()
    # plt.plot(data)
    # plt.xticks(rotation=30)
    # plt.show()

    '''
    SVR part
    '''
    x_data, y_data = preprocessing.mkData4SVR(data)
    svr = SVRmodel()
    svr.set_model(kernel='rbf')
    svr.get_data(x_data, y_data)
    svr.fit_model(x_data, y_data)
    svr.drawGraph()

    # svr_result = svr.predict_model(x_data)

    '''
    ARIMA model
    '''
    # Sationarity check with moving average
    rolling_data_mean, rolling_dataa_std = preprocessing.getMovingAverage(data,24)

    graph = Graphs()
    graph.plotBasicGraph(data, rolling_data_mean, rolling_dataa_std)

    # Sationarity check with adf function
    tsa = TimeSeriesAnalysis()
    tsa.adfFunction(data)

    # Diff for seasonality
    # loged_data = preprocessing.getLogedData(data)
    shifted_data = preprocessing.getShiftedData(data,24)
    diffed_data = preprocessing.getDiffData(data,shifted_data)

    # Sationarity check diffed data with moving average
    rolling_data_mean, rolling_dataa_std = preprocessing.getMovingAverage(diffed_data,24)
    # graph.plotBasicGraph(diffed_data, rolling_data_mean, rolling_dataa_std)

    # Sationarity check diffed data with adf function
    tsa = TimeSeriesAnalysis()
    tsa.adfFunction(diffed_data)

    # # Draw acf, pacf plot
    tsa.draw_acf(data)
    tsa.draw_pacf(data)
    # tsa.draw_acf(diffed_data)
    # tsa.draw_pacf(diffed_data)

    # Make ARIMA model
    tsa.mkARIMA_model(data,2,0,1)
    # tsa.mkSARIMA_model(data, 2, 0, 1, 2 ,0,1,24)

    # Predict with ARIMA
    # tsa.predict_plot(len(data)-24*30,len(data)+24,alpha=float(0.05))    # 한 달 전 데이터~24 시간 예측 데이터

    # Plot Test Data V.S. Prediction Data
    Test_data = preprocessing.resampled_data_1hour['2017-09-16':'2017-10-01']
    # Predict = tsa.model_fit.predict(start= len(data)-24*15, end = len(data)+24) # ARIMA 용
    Predict = tsa.model_fit.predict(start = len(data)-24*15, end=len(data)+24)
    plt.figure()
    plt.plot(Test_data)
    plt.plot(Predict)
    plt.xticks(rotation=15)
    plt.legend(loc='best')
    plt.show()

    # Make SARIMA model
    # tsa.mkSARIMA_model(data,2,0,0,1,0,0,24)
    # result = tsa.model_fit.predict(start=len(data)-24*15,end=len(data)+24,dynamic=True)
    # plt.plot(result)

    '''
    Data detrend (y - SVR result)
    '''
    # # Make preprocessed data
    # data_valued = data.values
    # preprocessed_data = data_valued - svr_result
    # preprocessed_data = preprocessed_data.reshape(-1,1)
    # preprocessed_data = pd.DataFrame(preprocessed_data)
    #
    # # Check stationarity with graph
    # rolled_mean, rolled_std = preprocessing.getMovingAverage(preprocessed_data)
    # graph.plotBasicGraph(preprocessed_data, rolled_mean, rolled_std)
    #
    # # Check stationarity with adf fuction
    # tsa.adfFunction(preprocessed_data)
