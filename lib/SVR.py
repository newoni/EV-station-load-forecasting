from sklearn.svm import SVR
import matplotlib.pyplot as plt

class SVRmodel:
    def __init__(self):
        pass

    def set_model(self, kernel='rbf'):
        self.model =SVR()

    def get_data(self, input, output):
        self.input_data = input
        self.output_data = output

    def fit_model(self, input, output):
        self.model.fit(input, output)

    def predict_model(self,X):
        return self.model.predict(X)

    def drawGraph(self):
        plt.figure()
        plt.plot(self.input_data, self.model.predict(self.input_data),color='b')
        plt.show()