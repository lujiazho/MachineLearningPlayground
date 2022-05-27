#########################################
#         MLPlayground for Numpy        #
#      Machine Learning Techniques      #
#                 v1.0.0                #
#                                       #
#         Written by Lujia Zhong        #
#       https://lujiazho.github.io/     #
#              26 May 2022              #
#              MIT License              #
#########################################
import numpy as np

class NearestMean:
    def __init__(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)       # starts from 0
        
        self.c = len(np.unique(self.labels)) # num of classes
        self.sample_means = []
    
    def fit(self):
        self.sample_means = []
        for i in range(self.c):
            self.sample_means.append(np.mean(self.data[self.labels==i], axis=0))
        self.sample_means = np.array(self.sample_means)

    def predict(self, inputs):
        pred = []
        for i, xy in enumerate(inputs):
            my_eu = []
            for j in range(self.c):
                # or np.linalg.norm(xy - self.sample_means[j])
                my_eu.append(np.sqrt(np.sum(np.square(xy - self.sample_means[j]))))
            pred.append(np.argmin(my_eu))
        return np.array(pred)