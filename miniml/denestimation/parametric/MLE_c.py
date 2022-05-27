import numpy as np

class MLE_c:
    def __init__(self, data):
        self.data = np.array(data)
        self.dim = 1 if len(self.data.shape)==1 else len(self.data[0])
        self.N = len(self.data)
    
    def fit(self, distribution='Gaussian'):
        self.distribution = distribution  # distribution type
        
        if self.distribution == 'Gaussian':
            self.mu = np.sum(self.data, axis=0)/self.N
            self.sigma = np.sum(np.matmul((self.data-self.mu).reshape(-1,self.dim,1), 
                                     (self.data-self.mu).reshape(-1,1,self.dim)), axis=0)/self.N
    
    def predict(self, inputs):
        if self.distribution == 'Gaussian':
            div_part = 1/np.sqrt(((2*np.pi)**self.dim)*np.linalg.det(self.sigma))
            exp_part = np.exp(np.sum((-1/2)*np.matmul(inputs-self.mu, 
                                                      np.linalg.inv(self.sigma))*(inputs-self.mu), axis=1))
            return div_part*exp_part
