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

class PCA:
    def __init__(self, x):
        self.x = np.array(x)
        self.dim = 1 if len(x.shape)==1 else len(x[0])
    
    def fit(self, n_components=1):
        self.n_components = n_components
        # step 1: calculate mean
        m = np.mean(self.x, axis=0)
        # step 2: calculate sample covariance matrix
        cov = np.matmul((self.x - m).T, (self.x - m)) / len(self.x)
        # step 3: calculate eigens
        self.eigenvalues, self.normalized_eigenvectors = np.linalg.eig(cov)
        # step 4: find maximum eigenvalues
        max_index = np.argsort(self.eigenvalues)[::-1]
        keep = min(self.n_components, self.dim)
        self.index = max_index[:keep]
    
    def transform(self, inputs):
        return np.matmul(inputs, self.normalized_eigenvectors[:,self.index])