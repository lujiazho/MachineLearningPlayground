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

class KDE:
    def __init__(self, data):
        self.data = np.array(data)
    
    def fit(self, kernel='Gaussian', h=1):
        self.kernel = kernel  # kernel type
        self.h = h            # width of kernel
        
    def K(self, x, data):
        if self.kernel == 'Gaussian': # gaussian pdf
            mu, sigma = 0, 1
            dim_x = 1 if len(x.shape)==1 else len(x[0])
            # part 1
            div_part = (1/(sigma * np.sqrt(2 * np.pi)))**dim_x

            expand = np.hstack([x.reshape(-1, 1) if len(x.shape)==1 else x for _ in range(len(data))])
            diff_part = (expand - data.reshape(1,-1)).reshape(-1, len(data), dim_x) / self.h
            # part 2
            exp_part = np.exp( - (diff_part - mu)**2 / (2 * sigma**2))
            return div_part * exp_part
    
    def predict(self, inputs):
        pred = self.K(x=inputs, data=self.data)
        # this product suppose that all r.v.s are uncorrelated, so f(x,y,z) = f(x)*f(y)*f(z)
        return np.sum(np.product(pred, axis=2), axis=1)/(len(self.data)*self.h) # product first, then add