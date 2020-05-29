import matplotlib.pyplot as plt

class Graphs:
    def __init__(self):
        pass

    def plotBasicGraph(self,*data):
        plt.figure(figsize = (8,6))
        plt.xlabel('Time')
        plt.ylabel('Power')
        plt.xticks(rotation=30)
        plt.plot(data[0],color='black', label='Original')
        plt.plot(data[1],color='red', label="Mean")
        plt.plot(data[2],color='blue', label="Std")
        plt.legend(loc='best')
        # plt.title('Power Usage Profile')
        plt.show()