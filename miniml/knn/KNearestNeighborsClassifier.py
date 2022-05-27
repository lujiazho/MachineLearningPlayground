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

class KNearestNeighborsClassifier:
    def __init__(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)       # starts from 0
    
    def fit(self, k):
        self.k = k

    def self_cdist(self, inputs):
        dist = []
        dim = len(self.data[0])
        calc = np.hstack([inputs for _ in range(len(self.data))]) - self.data.reshape(1, -1)
        for i in range(len(self.data)):
            class_ = calc[:,i*dim:(i+1)*dim]
            dist.append(np.linalg.norm(class_, axis=1).reshape(-1,1))
        return np.hstack(dist)
    
    def getMode(self, x):
        vals, counts = np.unique(x, return_counts=True)
        index = np.argmax(counts)
        return vals[index]
    
    def predict(self, inputs):
        # step 1: calculate the dist between each point and each data in dataset
        dist_mat = self.self_cdist(inputs)
        
        # step 2: for each point, pick the nearest k data in dataset
        index = np.argsort(dist_mat, axis=1)[:,:self.k]
        
        # step 3: for each point, calculate assigned class by nearest k data's classes
        preds = list(map(self.getMode, self.labels.reshape(-1,1)[index].reshape(-1,self.k)))
        return np.array(preds)