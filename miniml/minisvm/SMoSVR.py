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

class SMoSVR:
    def __init__(self, x=[], y=[]):
        self.x = np.array(x)          # data
        self.y = np.array(y)
        self.N = len(x)               # number of points
        
        self.beta = []

        self.w = []                   # optimal w
        self.w0 = None                # optimal bias term w0

    # n should be not equal to m
    def random_n(self, m):
        n = random.randint(0, self.N-1)
        while n == m:
            n = random.randint(0, self.N-1)
        return n

    def K_(self, x):
        if self.kernel == 'RBF':
            return np.exp(-self.gamma*((np.sum(self.x*self.x, axis=1) - 2*np.matmul(self.x, x.T).T).T + np.sum(x*x, axis=len(x.shape)-1)))
        if self.kernel == 'polynomial':
            return (self.gamma*np.matmul(self.x, x.T)+self.r)**self.d
        return np.matmul(self.x, x.T)
    
    def K_ij(self, i, j):
        if self.kernel == 'RBF': # RBF kernel's Φ(x) is infinite dimensional
            return np.exp(-self.gamma*np.sum((self.x[i]-self.x[j])*(self.x[i]-self.x[j])))
        if self.kernel == 'polynomial':
            return (self.gamma*np.dot(self.x[i], self.x[j])+self.r)**self.d
        return np.dot(self.x[i], self.x[j])
    
    def f_k(self, k):
        return np.sum((self.beta*self.K_(self.x[k]).T).T, axis=0) + self.w0
    
    def sign(self, x):
        return x/abs(x) if x else 0
    
    def update_w0(self, idx, m, n, o_m, n_m, o_n, n_n):
        return self.w0 + self.y[idx] - self.f_k(idx) + (o_m-n_m)*self.K_ij(m,idx) + (o_n-n_n)*self.K_ij(idx,n)

    def fit(self, max_epoch=100, kernel='linear', C=1.0, epsilon=0.1, gamma=1, d=2, r=1):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.d = d
        self.r = r

        self.beta = np.zeros((self.N))         # initialize beta
        self.w0 = 0                            # initialize w0
        while max_epoch:
            prev_bata = np.copy(self.beta)
            for m in range(self.N):
                # step 1: randomly choose n != m
                n = self.random_n(m)
                
                old_n, old_m = self.beta[n], self.beta[m]
                
                # step 2: calculate temporary beta_n
                alpha = old_n + old_m
                eta = self.K_ij(m,m) + self.K_ij(n,n) - 2*self.K_ij(m,n)
                if not eta: # in case there are two points have exactly the same coordinates
                    continue
                cort = 2*epsilon/eta
                temp_n = old_n + (self.y[n]-self.y[m]+self.f_k(m)-self.f_k(n))/eta
                temp_m = alpha - temp_n
                
                # step 3: apply correction to temporary beta_n
                if temp_m*temp_n < 0:
                    if abs(temp_m) >= cort and abs(temp_n) >= cort:
                        temp_n += self.sign(temp_m)*cort
                    else:
                        temp_n = alpha if abs(temp_n) > abs(temp_m) else 0
                
                # step 4: crop temporary beta_n
                L = max(alpha-self.C, -self.C)
                H = min(self.C, alpha+self.C)
                beta_n = min(max(temp_n, L), H)
                
                # step 5: calculate new beta_m
                beta_m = alpha - beta_n
                
                # step 6: calculate new w0 (slack form)
                if not beta_m:
                    w0 = self.update_w0(m, m, n, old_m, beta_m, old_n, beta_n)
                elif not beta_n:
                    w0 = self.update_w0(n, m, n, old_m, beta_m, old_n, beta_n)
                else:
                    w0_1 = self.update_w0(m, m, n, old_m, beta_m, old_n, beta_n)
                    w0_2 = self.update_w0(n, m, n, old_m, beta_m, old_n, beta_n)
                    w0 = (w0_1+w0_2)/2
                
                # update all together
                self.beta[m] = beta_m
                self.beta[n] = beta_n
                self.w0 = w0
            
            max_epoch -= 1
            if max_epoch % 100 == 0 and max_epoch:
                print(f"Last {max_epoch} Epochs")
        return [max_epoch, C, epsilon]
    
    def predict(self, x):
        result = np.sum((self.beta*self.K_(x).T).T, axis=0) + self.w0
        return result