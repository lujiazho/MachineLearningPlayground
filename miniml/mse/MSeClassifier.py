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

class MSeClassifier:
    def __init__(self, x=[], z=[], b=[]):
        self.x = np.array(x)          # data
        self.z = np.array(z)          # label
        self.b = np.array(b)
        self.N = len(x)               # number of points

        self.w = []                   # augmented w, which includes w0, w1, ...

    def algebraicSol(self):
        pseudo = np.linalg.inv(np.matmul(self.zx.T, self.zx))
        self.w = np.matmul(np.matmul(pseudo, self.zx.T), self.b)
        
    def J_n(self, n):
        return (2/self.N)*(np.matmul(self.w, self.x[n])-self.z[n]*self.b[n])*self.x[n]
    
    def seqGDSol(self):
        while self.max_epoch:
            for n in range(self.N):
                self.w = self.w - self.lr*self.J_n(n)
            
            self.max_epoch -= 1
            if self.max_epoch % 5000 == 0 and self.max_epoch:
                print(f"Last {self.max_epoch} Epochs")
                
    def kernelTransform(self, x):
        '''Only for two dimension data'''
        if self.kernel == 'quadratic':
            return np.array(list(map(lambda x: [1, x[1], x[2], x[1]*x[2], x[1]*x[1], x[2]*x[2]], x)))
        return x
        
    def fit(self, max_epoch=100, sol='algebra', kernel='linear', lr=0.01):
        self.sol = sol
        self.max_epoch = max_epoch
        self.kernel = kernel
        self.lr = lr
        
        if self.kernel != 'linear':
            self.x = self.kernelTransform(self.x)
        
        self.w = np.zeros(len(self.x[0]))        # initialize w
        
        if self.sol == 'algebra':
            self.zx = np.array([self.z[i]*per for i, per in enumerate(self.x)])
            self.algebraicSol()
        elif self.sol == 'GD':
            self.seqGDSol()
        
        return [max_epoch, self.sol, self.lr]
    
    def predict(self, x):
        x = np.insert(x, 0, [1 for _ in range(len(x))], axis=1) # augmented notation
        if self.kernel != 'linear':
            x = self.kernelTransform(x)
        
        result = np.matmul(x, self.w)
        result[result>0] = 0
        result[result<0] = 1
        return result