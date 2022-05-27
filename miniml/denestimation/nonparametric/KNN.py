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

class KNN:
    def __init__(self, data):
        self.data = np.array(data)
    
    def fit(self, k=3):
        self.k = k
    
    def self_cdist(self, inputs):
        dist = []
        dim = len(self.data[0])
        calc = np.hstack([inputs for _ in range(len(self.data))]) - self.data.reshape(1, -1)
        for i in range(len(self.data)):
            class_ = calc[:,i*dim:(i+1)*dim]
            dist.append(np.linalg.norm(class_, axis=1).reshape(-1,1))
        return np.hstack(dist)
    
    def predict(self, inputs):
        # step 1: calculate the dist between each point and each data in dataset
        dist_mat = self.self_cdist(inputs)
        
        # step 2: for each point, pick the farest kth data in dataset
        index = np.argsort(dist_mat, axis=1)[:, self.k-1]

        # step 3: for each point, calculate predicted probability
        area = np.pi*np.linalg.norm(inputs - self.data[index], axis=1)**2
        
        return (self.k/len(self.data))/area