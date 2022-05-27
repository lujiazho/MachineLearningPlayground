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

class KNearestNeighborsRegressor:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
    
    def fit(self, k):
        self.k = k

    def self_cdist(self, inputs):
        dist = []
        dim = len(self.x[0])
        calc = np.hstack([inputs for _ in range(len(self.x))]) - self.x.reshape(1, -1)
        for i in range(len(self.x)):
            class_ = calc[:,i*dim:(i+1)*dim]
            dist.append(np.linalg.norm(class_, axis=1).reshape(-1,1))
        return np.hstack(dist)
    
    def predict(self, inputs, weights='uniform'):        
        # step 1: calculate the dist between each point and each data in dataset
        dist_mat = self.self_cdist(inputs)
        
        # step 2: for each point, pick the nearest k data in dataset
        index = np.argsort(dist_mat, axis=1)[:,:self.k]

        # step 3: for each point, calculate y vals by nearest k data's y's
        vals = self.y.reshape(-1,1)[index].reshape(-1,self.k)
        if weights == 'uniform':
            preds = np.mean(vals, axis=1)
        elif weights == 'distance':
            ws = 1/(np.array(list(map(sorted, dist_mat)))[:,:self.k]+0.1) # weights
            preds = np.zeros(vals[:,0].shape).reshape(-1,1)
            for i in range(self.k):
                preds += (vals[:,i]*ws[:,i]).reshape(-1,1)
            preds /= np.sum(ws, axis=1).reshape(-1,1)
        else:
            return None
        return preds