import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

class SVM(object):
    def __init__(self,visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

    def fit(self, data):
        
        self.data = data
        opt_dict = {}
        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]


if __name__ == "__main__":

    data_dict = {-1:np.array([[1,7],[2,8],[3,8]]),1:np.array([[5,1],[6,-1],[7,3]])}

    #print(data_dict)

    svm = SVM()
    svm.fit(data = data_dict)
