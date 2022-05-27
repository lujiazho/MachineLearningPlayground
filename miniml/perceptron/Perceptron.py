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

class Perceptron:
    def __init__(self, data, labels):
        self.data = np.array(data)     # augmented data
        self.labels = np.array(labels) # starts from 0
        
        self.w = []                    # augmented notation

    # Initialize indicator function
    def indicator(self, x):
        if type(x) != bool and type(x) != np.bool_:
            return x*np.full(len(x), 1)
        return 1 if x == True else 0
    
    # criterion function
    def J(self, trains, labels, ind=0):
        if ind == 1:
            print(self.w)
        z = np.full(len(trains), 1)
        z[labels != 0] = -1
        com = np.matmul(trains, self.w)*z
        return -np.sum(self.indicator(com<=0)*com)
    
    # Derivative of sequential criterion function
    def J_i_derivative(self, x, label):
        z = 1 if label == 0 else -1
        return -self.indicator(np.sum(self.w*z*x)<=0)*z*x
    
    def J_batch_derivative(self, data, labels):
        z = np.copy(labels)
        z[labels==0] = 1
        z[labels!=0] = -1
        return -np.sum((self.indicator(np.matmul(data, self.w)*z<=0)*z).reshape(-1,1)*data, axis=0)
    
    # theoretically, exp, sin can also be used, but they perform wrose when it comes to classification
    def nonlinearTransform(self, x):
        if self.kernel == 'quadratic':
            return np.array(list(map(lambda x: [1, x[1], x[2], x[1]*x[2], x[1]*x[1], x[2]*x[2]], x)))
        return x
    
    # SequentialGD
    def fit(self, iters = 10000, kernel='linear', GD='SGD'):
        n = len(self.data) # total num of data
        self.kernel = kernel
        
        if self.kernel != 'linear':
            self.data = self.nonlinearTransform(self.data)
        
        # step 1: Shuffle dataset and corresponding labels
        indices = list(range(n))
        random.shuffle(indices)
        trains = self.data[indices]
        labels = self.labels[indices]
        
        # step 2: Initialize w, lr
        self.w = np.full(len(self.data[0]), 0.1)
        lr = 0.0001
        
        smallest_J = float("inf")
        best_W = np.copy(self.w)

        start = 0
        size = 5
        losses = []
        loss_light = []
        # step 3: sequential gradient descent
        while (iters):
            if GD == 'SGD':
                self.w -= lr*self.J_i_derivative(trains[iters%n], labels[iters%n])
            elif GD == 'MBGD': # mini batch
                if start + size < n:
                    end = start + size
                else:
                    start = 0
                    end = start + size
                self.w -= lr*self.J_batch_derivative(trains[start:end], labels[start:end])
            elif GD == 'BGD': # batch GD
                self.w -= lr*self.J_batch_derivative(trains, labels)
            cur_J = self.J(trains, labels)
            losses.append(cur_J)
            if not cur_J:
                print("Data is linearly separable")
                loss_light.append(1)
                break
            iters -= 1
            if (cur_J < smallest_J):
                smallest_J = cur_J
                best_W = np.copy(self.w)
                loss_light.append(1)
            else:
                loss_light.append(0)
        
        self.w = best_W
        return [iters, losses, loss_light]
        
    def predict(self, x):
        x = np.insert(x, 0, [1 for _ in range(len(x))], axis=1) # augmented notation
        if self.kernel != 'linear':
            x = self.nonlinearTransform(x)
        
        pred = np.matmul(x, self.w)
        pred[pred>0] = 0 # class 1
        pred[pred<0] = 1 # class 2
        return pred