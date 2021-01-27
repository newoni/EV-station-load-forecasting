# EV-station-load-forecasting

## main

## lib\\graphs.py
#### Plot 3 types of data. Original data, Mean, Standard deviation
#### Used matplotlib.pyplot

## lib\\preprocessing.py
#### preprocess data before insert SVR, TSA model
#### resampling or other mathmatical preprocessing(handle log, differentiation etc.) 
#### Used pandas, pickle, numpy, datetime (mainly used pandas)

## lib\\SVR.py
#### Used sklearn's SVR model

## lib\\time_series_analysis.py
#### Used statsmodel's time series analysis mdoel ARIMA, SARIMAX
#### checked data with acf, pacf, dicky fuller test before conclude ARIMA SARIMAX model
