import pandas as pd
import datetime as dt
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lib.preprocessing import Preprocession4TimeSeriesAnalysis
from lib.time_series_analysis import TimeSeriesAnalysis
from lib.graphs import Graphs
from lib.SVR import SVRmodel

if __name__ =="__main__":
    '''
    Preprocessing Data
    '''

    # Data load
    preprocessing = Preprocession4TimeSeriesAnalysis()

    # Data preprocessing
    #preprocessing.iter_oper()      # preprocess the data

    # Load preprocessed data
    preprocessing.load_preprocessed_data()
    data = preprocessing.resampled_data_1hour

    # Select the horizon of data
    # data = preprocessing.cutDataOnTime(data, '2017-01-01', '2017-01-03')

    '''
    SVR part
    '''
    x_data, y_data = preprocessing.mkData4SVR(data)
    svr = SVRmodel()
    svr.set_model(kernel='rbf')
    svr.get_data(x_data, y_data)
    svr.fit_model(x_data, y_data)
    svr.drawGraph()

    svr_result = svr.predict_model(x_data)

    '''
    ARIMA model
    '''
    # Sationarity check with moving average
    rolling_data_mean, rolling_dataa_std = preprocessing.getMovingAverage(data)

    graph = Graphs()
    graph.plotBasicGraph(data, rolling_data_mean, rolling_dataa_std)

    # Sationarity check with adf function
    tsa = TimeSeriesAnalysis()
    tsa.adfFunction(data)

    # Diff for sationarity
    # loged_data = preprocessing.getLogedData(data)
    # shifted_data = preprocessing.getShiftedData(loged_data,24)
    # diffed_data = preprocessing.getDiffData(loged_data,shifted_data)
    # data = diffed_data

    # Sationarity check diffed data with moving average
    # rolling_data_mean, rolling_dataa_std = preprocessing.getMovingAverage(loged_data)
    #
    # data = loged_data - rolling_data_mean
    # data.dropna(inplace=True)
    #
    # rolling_data_mean, rolling_dataa_std = preprocessing.getMovingAverage(data)
    #
    # graph = Graphs()
    # graph.plotBasicGraph(data, rolling_data_mean, rolling_dataa_std)

    # Sationarity check diffed data with adf function
    # tsa = TimeSeriesAnalysis()
    # tsa.adfFunction(data)
    #
    # # Draw acf, pacf plot
    tsa.draw_acf(data)
    tsa.draw_pacf(data)

    # Make ARIMA model
    tsa.mkARIMA_model(data, 2, 0, 2)

    # Predict with ARIMA
    tsa.predict_plot(len(preprocessing.resampled_data_1hour)-24*30,len(preprocessing.resampled_data_1hour)+24,alpha=float(0.05))    # 한 달 전 데이터~24 시간 예측 데이터
    # tsa.predict_plot(1,len(preprocessing.resampled_data_1hour)+24,alpha=float(0.05))

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
