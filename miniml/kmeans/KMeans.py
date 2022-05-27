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
import random
import numpy as np

class KMeans:
    def __init__(self, data):
        self.data = np.array(data)
    
    def self_cdist(self):
        dis = []
        for i in range(self.k):
            class_ = (np.hstack([self.data for _ in range(self.k)]) - self.means.reshape(1, -1))[:,i*self.dim:(i+1)*self.dim]
            dis.append(np.linalg.norm(class_, axis=1).reshape(-1,1))
        return np.hstack(dis)
    
    def fit(self, k=3):
        if not len(self.data):
            return
        self.dim = len(self.data[0])  # dimension of data points
        self.k = k
        self.means = []
        
        # step 1: initialize k cluster centers with the same dimension as each point
        for _ in range(self.k):
            self.means.append(random.choice(self.data))
        self.means = np.array(self.means)

        iteration = 100
        while iteration:
            # step 2: find the nearest cluster center for each point
            dist_mat = self.self_cdist()
            new_class = np.argmin(dist_mat, axis=1)
            
            # step 3: recalculate cluster center for each cluster corresponding new points assignment
            new_means = []
            for i in range(self.k):
                new_means.append(np.mean(self.data[new_class==i], axis=0))
            new_means = np.array(new_means)
            
            # step 4: if means didn't change, algorithm stops
            if np.sum(new_means == self.means) == self.dim*self.k:
                break
            self.means = new_means
            iteration -= 1
        return [iteration, self.k]

    def predict(self, inputs):
        pred = []
        for i, xy in enumerate(inputs):
            my_eu = []
            for j in range(self.c):
                # or np.linalg.norm(xy - self.sample_means[j])
                my_eu.append(np.sqrt(np.sum(np.square(xy - self.means[j]))))
            pred.append(np.argmin(my_eu))
        return np.array(pred)