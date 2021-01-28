import pandas as pd
import datetime as dt
import pickle
import numpy as np

class Preprocessing4TimeSeriesAnalysis:
    def __init__(self):
        pass

    def read_data(self):
        print("**** 데이터 import 시작 ****")
        self.data = pd.read_csv('data\\electric-chargepoint-analysis-2017-raw-domestics-data.csv')
        print("**** 데이터 import 완료 ****")

    def drop_unnecessary_column(self):
        print("**** 불필요 column 제거 시작 ****")
        self.data.drop("PluginDuration",axis=1,inplace=True)
        self.data.drop("ChargingEvent",axis=1,inplace=True)
        self.data.drop("CPID",axis=1,inplace=True)
        print("**** 불필요 column 제거 완료 ****")

    def change_str2dt(self):
        print("**** Start Change String to Date Time ****")
        date_list = []
        for i in range(len(self.data)):
            if i%100000 ==0:
                print("****",i,"번째 진행중, 진행률:", round(i/len(self.data)*100,2),"% ****")
            year, month, day = self.data["StartDate"][i].split("-")
            hour, minute, second = self.data["StartTime"][i].split(":")

            date = dt.datetime(int(year),int(month),int(day),int(hour),int(minute),int(second))

            date_list.append(date)

        self.data["DateTime"] = np.array(date_list)

        print("**** Changing String to Date Time finished ****")

    def rename_index_name(self):
        print("**** index rename 시작 ****")
        self.data = self.data.rename(self.data["DateTime"])
        print("**** index rename 완료 ****")

    def resample_1hour(self):
        print("**** 1시간 단위 resample 시작 ****")
        self.resampled_data_1hour = self.data["Energy"].resample("1H").sum()
        print("**** 1시간 단위 resample 완료 ****")

    def resample_1day(self):
        print("**** 1일 단위 resample 시작 ****")
        self.resampled_data_1day = self.data["Energy"].resample("1D").sum()
        print("**** 1일 단위 resample 완료 ****")


    def save_as_pickle(self):
        print("**** Save preprocessed data ****")
        with open('pickle\\resampled_data_1hour.pickle', 'wb') as fr:
            pickle.dump(self.resampled_data_1hour, fr)
        with open('pickle\\resampled_data_1day.pickle', 'wb') as fr:
            pickle.dump(self.resampled_data_1day, fr)
        print("**** Complete saving preprocessed data ****")

    # Data preprocessing
    def iter_oper(self):
        self.read_data()
        self.drop_unnecessary_column()
        self.change_str2dt()
        self.rename_index_name()
        self.resample_1hour()
        self.resample_1day()
        self.save_as_pickle()
        return self.resampled_data_1hour

    # after using iter_oper, load_preprocessed_data
    def load_preprocessed_data(self):
        with open("pickle\\resampled_data_1hour.pickle", "rb") as fr:
            self.resampled_data_1hour = pickle.load(fr)

        with open("pickle\\resampled_data_1day.pickle", "rb") as fr:
            self.resampled_data_1day = pickle.load(fr)

    def cutDataOnTime(self, data, start, end):
        self.cutted_data = data[start:end]
        return self.cutted_data

    def getMovingAverage(self,data,rolling_number):
        self.rolling_data_mean = data.rolling(rolling_number).mean()
        # self.rolling_data_mean.dropna(inplace=True)

        self.rolling_data_std = data.rolling(rolling_number).std()
        # self.rolling_data_std.dropna(inplace=True)

        return self.rolling_data_mean, self.rolling_data_std

    def getShiftedData(self, data, shifting_num):
        self.shifted_data = data.shift(shifting_num)
        return self.shifted_data

    def getLogedData(self,data):
        self.logged_data = np.log(data)
        self.logged_data = self.logged_data.replace(-np.inf,np.nan)
        self.logged_data.dropna(inplace=True)
        return self.logged_data

    def getDiffData(self, data1, data2):
        self.diffed_data = data1 - data2
        self.diffed_data.dropna(inplace=True)
        return self.diffed_data

    def mkData4SVR(self, data):
        self.X_data = np.arange(len(data))
        self.X_data = self.X_data.reshape(-1,1)

        self.Y_data = np.array(data)
        self.Y_data = self.Y_data.reshape(-1,1)

        return self.X_data, self.Y_data