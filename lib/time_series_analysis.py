from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

class TimeSeriesAnalysis:
    def __init__(self):
        pass

    def adfFunction(self,data):
        self.adffuller_result=adfuller(data)

        print("ADF Statistic: {}".format(self.adffuller_result[0]))
        print("P-value: {}".format(self.adffuller_result[1]))
        print("Critical Values:")
        for key, value in self.adffuller_result[4].items():
            print('\t{}:{}'.format(key, value))

    def draw_acf(self, data):
        plot_acf(data)

    def draw_pacf(self, data):
        plot_pacf(data)

    def mkARIMA_model(self,data, p,d,q):
        self.model = ARIMA(data,order=(p,d,q))
        self.model_fit = self.model.fit(disp=-1)
        print(self.model_fit.summary())

    def mkSARIMA_model(self,data,p,d,q, P,D,Q,time_step):
        self.model = SARIMAX(data,order=(p,d,q), seasonal_order=(P,D,Q,time_step))
        self.model_fit= self.model.fit(disp=1)
        print(self.model_fit.summary())

    def predict_plot(self,start,end, alpha=0.05):   # arima model 일 경우 사용 sarima일 경우 직접 사용이 편함.
        self.model_fit.plot_predict(start,end,alpha=alpha)


