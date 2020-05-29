from statsmodels.tsa.arima_model import ARIMA
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

    def predict_plot(self,start,end, alpha=0.05):
        self.model_fit.plot_predict(start,end,alpha=alpha)


