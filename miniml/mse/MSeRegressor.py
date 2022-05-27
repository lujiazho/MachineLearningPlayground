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

class MSeRegressor:
    def __init__(self, x=[], y=[]):
        self.x = np.array(x)          # data
        self.y = np.array(y)
        self.N = len(x)               # number of points
        
        self.w = []                   # augmented w, which includes w0, w1, ...

    # OLS
    def algebraicSol(self):
        pseudo = np.linalg.inv(np.matmul(self.x.T, self.x))
        self.w = np.matmul(np.matmul(pseudo, self.x.T), self.y)
        
    def J_n(self, n):
        return (2/self.N)*(np.matmul(self.w, self.x[n])-self.y[n])*self.x[n]
    
    # LMS
    def seqGDSol(self):
        while self.max_epoch:
            for n in range(self.N):
                self.w = self.w - self.lr*self.J_n(n)
            
            self.max_epoch -= 1
            if self.max_epoch % 5000 == 0 and self.max_epoch:
                print(f"Last {self.max_epoch} Epochs")
                
    # generally many function can be used as transformation: exp, sin, cos, 1/x, ...
    def kernelTransform(self, x):
        '''Only for one dimension data'''
        if self.kernel == 'polynomial':
            gen = np.hstack([np.reshape(x[:,1]**i, (-1, 1)) for i in range(2, self.d+1)])
            return np.hstack((x, gen))
        if self.kernel == 'sin':
            return np.sin(x)
        return x
        
    def fit(self, max_epoch=100, sol='algebra', kernel='linear', lr=0.01, d=3):
        self.sol = sol
        self.max_epoch = max_epoch
        self.kernel = kernel
        self.lr = lr
        self.d = d
        
        if self.kernel != 'linear':
            self.x = self.kernelTransform(self.x)
        
        self.w = np.zeros(len(self.x[0]))        # initialize w
        
        if self.sol == 'algebra':
            self.algebraicSol()
        elif self.sol == 'GD':
            self.seqGDSol()
        
        return [max_epoch, self.sol, self.lr]
    
    def predict(self, x):
        if self.kernel != 'linear':
            x = self.kernelTransform(x)
        result = np.matmul(x, self.w)
        return result