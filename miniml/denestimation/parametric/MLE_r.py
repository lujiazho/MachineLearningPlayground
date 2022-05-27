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

class MLE_r:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = y
        
        self.dim = 1 if len(self.x.shape)==1 else len(self.x[0])
        self.N = len(self.x)
        
    # generally many function can be used as transformation: exp, sin, cos, 1/x, ...
    def kernelTransform(self, x):
        if self.kernel == 'sin':
            return np.sin(x)
        return x
    
    def fit(self, distribution='Gaussian', kernel='linear'):
        self.distribution = distribution  # distribution type
        self.kernel = kernel
        
        if self.kernel != 'linear':
            self.x = self.kernelTransform(self.x)
        
        # MLE regression with Gaussian has the same form as OLS
        if self.distribution == 'Gaussian':
            self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.x.T, self.x)), self.x.T), self.y)
            self.sigma = np.sqrt(np.sum((self.y-np.matmul(self.x, self.w))**2)/self.N)
    
    def predict(self, inputs, y):
        if self.kernel != 'linear':
            inputs = self.kernelTransform(inputs)
        
        if self.distribution == 'Gaussian':
            div_part = 1/(np.sqrt(2*np.pi)*self.sigma)
            exp_part = np.exp((-1/2)*((y-np.matmul(inputs, self.w))/self.sigma)**2)
            return div_part*exp_part