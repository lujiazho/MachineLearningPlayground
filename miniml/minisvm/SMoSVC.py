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
from .Plot import *
import matplotlib.animation as animation

class SMoSVC(Plot):
    def __init__(self, z=[], u=[], b=1, val=[], val_z=[]):
        self.z = np.array(z)          # classes, only for two: 1 or -1
        self.u = np.array(u)          # data
        self.N = len(z)               # number of points

        # validation data
        self.val = np.array(val)
        self.val_z = np.array(val_z)
        
        # parameters for calc lambda
        self.b = b                    # margin
        self.lambda_ = []             # lambda
        
        # parameters for calc w
        self.w = []                   # optimal w
        self.w0 = None                # optimal bias term w0
        
        # kernel types
        self.s_kernels = ['linear']
        self.c_kernels = ['RBF', 'quadratic', 'polynomial', 'sigmoid']

    # n should be not equal to m
    def random_n(self, m):
        n = random.randint(0, self.N-1)
        while n == m:
            n = random.randint(0, self.N-1)
        return n
    
    # for complex kernel
    def K_ij(self, i, j):
        if self.kernel == 'RBF': # RBF kernel's Φ(x) is infinite dimensional
            return np.exp(-self.gamma*np.sum((self.u[i]-self.u[j])*(self.u[i]-self.u[j])))
        if self.kernel == 'quadratic':
            return (self.gamma*np.dot(self.u[i], self.u[j])+self.r)**2
        if self.kernel == 'polynomial':
            return (self.gamma*np.dot(self.u[i], self.u[j])+self.r)**self.d
        if self.kernel == 'sigmoid':
            return np.tanh(self.gamma*np.dot(self.u[i], self.u[j])+self.r)
        return np.dot(self.u[i], self.u[j])
    
    # matrix version of K_ij for accelerating computation, x could be one or two dim
    def K_(self, x):
        if self.kernel == 'RBF':
            return np.exp(-self.gamma*((np.sum(self.u*self.u, axis=1) - 2*np.matmul(self.u, x.T).T).T + np.sum(x*x, axis=len(x.shape)-1)))
        if self.kernel == 'quadratic':
            return (self.gamma*np.matmul(self.u, x.T)+self.r)**2
        if self.kernel == 'polynomial':
            return (self.gamma*np.matmul(self.u, x.T)+self.r)**self.d
        if self.kernel == 'sigmoid':
            return np.tanh(self.gamma*np.matmul(self.u, x.T)+self.r)
        return np.matmul(self.u, x.T)
    
    def f_k(self, k):
        return np.sum((self.lambda_*self.z*self.K_(self.u[k]).T).T, axis=0) + self.w0
    
    def E_k(self, k):
        return self.f_k(k) - self.z[k]*self.b
    
    def crop_lambda_n(self, m, n, lambda_n):
        old_n, old_m = self.lambda_[n], self.lambda_[m]
        if self.z[m] != self.z[n]:
            return min(max(max(0, old_n - old_m), lambda_n), min(self.C, self.C + old_n - old_m))
        else:
            return min(max(max(0, old_m + old_n - self.C), lambda_n), min(old_m + old_n, self.C))
    
    def update_w(self):
        self.w = np.sum(np.multiply(self.lambda_*self.z, self.u.T).T, axis=0)
        
    def update_w0_1(self, E, m, n, o_m, n_m, o_n, n_n):
        return self.w0 - E + self.z[m]*self.K_ij(m,m)*(o_m-n_m) + self.z[n]*self.K_ij(n,m)*(o_n-n_n)
    
    def update_w0_2(self, E, m, n, o_m, n_m, o_n, n_n):
        return self.w0 - E + self.z[m]*self.K_ij(m,n)*(o_m-n_m) + self.z[n]*self.K_ij(n,n)*(o_n-n_n)
    
    # deprecated
    # def k_quadratic(self, x):
    #     return np.array(list(map(lambda x: [x[0],x[1],x[0]*x[0], x[1]*x[1], x[0]*x[1]], x)))
    
    # only for kernel that can be easily stated as K(x,y) = Φ(x)*Φ(y)
    def apply_kernel(self):
        if self.kernel == 'linear':
            return
        # deprecated
        # if self.kernel == 'quadratic':
        #     self.u = self.k_quadratic(self.u)
    
    def fit(self, max_epoch=100, kernel='linear', C=1.0, epsilon=0.0001, ani=False, gamma=3, d=3, r=1):
        """
        Args:
            max_epoch: Maximum training epoch if linear KKT cannot be satisfied.
            kernel: Kernel function. Default is linear.
            C: Upper Limit of lambda. A parameter of soft margin/slack-SVM.
            epsilon: Threshold of lambda value change, if which is less than this & KKT is satisfied, then traning ends in advance.
            ani: Launch traning process record as .gif file. This may cause much slower traning because of drawing in each epoch.
            gamma: Parameter of RBF/quadratic/poly/sigmoid kernels.
            d: Parameter of polynomial kernel.
            r: Parameter of poly/sigmoid kernels.

        Returns:
            Training Parameters & results: [max_epoch, kernel, C, epsilon, ani, gamma, d, acc_train, acc_val]
        """
        self.kernel = kernel
        self.C = C
        self.ani = ani
        self.gamma = gamma
        self.d = d
        self.r = r

        if kernel not in self.s_kernels and kernel not in self.c_kernels:
            print("[Error] No such kernel type!")
            return
        self.apply_kernel()                    # check for simple kernel
        
        self.ims = []                          # for ani

        val_accs = [0]
        max_acc = 0                        # record the maximum and store it's lambda and w0
        self.best_lamd = np.zeros((self.N))
        self.best_w0 = 0
        
        self.lambda_ = np.zeros((self.N))      # initialize lambda
        self.w0 = 0                            # initialize w0
        while max_epoch:
            prev_lambda = np.copy(self.lambda_)
            for m in range(self.N):
                # step 1: randomly choose n != m
                n = self.random_n(m)
                
                old_n, old_m = self.lambda_[n], self.lambda_[m]
                
                # step 2: calculate new lambda_n
                Em, En = self.E_k(m), self.E_k(n)
                eta = self.K_ij(m,m) + self.K_ij(n,n) - 2*self.K_ij(m,n)
                if not eta: # in case there are two points have exactly the same coordinates
                    continue
                lambda_n = old_n + self.z[n]*(Em-En)/eta
                
                # step 3: crop new lambda_n
                lambda_n = self.crop_lambda_n(m, n, lambda_n)
                # if lambda_n < 1e-9, it should be zero because it's not support vector. So we skip it
                # if we don't, the little change of λn can cause big change of result, leading to wrong classification
                if lambda_n < 1e-9:
                    continue
                
                # step 4: calculate new lambda_m
                lambda_m = old_m - self.z[m]*self.z[n]*(lambda_n - old_n)
                
                # step 5: calculate new w0 (slack form)
                if self.C > lambda_m > 0:
                    w0 = self.update_w0_1(Em, m, n, old_m, lambda_m, old_n, lambda_n)
                elif self.C > lambda_n > 0:
                    w0 = self.update_w0_2(En, m, n, old_m, lambda_m, old_n, lambda_n)
                else:
                    cond = (self.lambda_>1e-5)&(self.lambda_<self.C)
                    lam = self.lambda_[cond]
                    if len(lam):
                        w0 = self.b*self.z[cond][0] - np.sum((self.lambda_*self.z*self.K_(self.u[cond][0]).T).T, axis=0)
                    else: # No available point for W0, we adopt no slack case form
                        w0_1 = self.update_w0_1(Em, m, n, old_m, lambda_m, old_n, lambda_n)
                        w0_2 = self.update_w0_2(En, m, n, old_m, lambda_m, old_n, lambda_n)
                        w0 = (w0_1 + w0_2)/2
                
                # update all together
                self.lambda_[m] = lambda_m
                self.lambda_[n] = lambda_n
                self.w0 = w0
            
            # use val data to test model each epoch and store the best lambda and w0
            if self.val.shape[0] and self.val_z.shape[0]:
                cur_res = np.sum((self.lambda_*self.z*self.K_(self.val).T).T, axis=0) + self.w0
                cur_res[cur_res>0] = 1
                cur_res[cur_res<0] = -1

                cur_acc = round(np.sum(cur_res == self.val_z)/len(self.val_z), 5)
                val_accs.append(cur_acc)
                if cur_acc > max_acc:
                    max_acc = cur_acc
                    self.best_lamd = np.copy(self.lambda_)
                    self.best_w0 = self.w0

            # we don't update w, just use kernel for everywhere w is needed
            if self.kernel == 'linear':
                self.update_w() # for plot margin in 2D
    
            if self.ani:
                self.plot()
            
            # check linearly separable case (no slack) KKT
            if np.linalg.norm(prev_lambda - self.lambda_) < epsilon and self.check_KKT_lambda() and self.check_KKT_w_w0():
                print(f"KKT Satisfied!")
                break
            
            max_epoch -= 1
            if max_epoch % 100 == 0 and max_epoch:
                print(f"Last {max_epoch} Epochs")
        if self.ani:
            ani = animation.ArtistAnimation(self.fig, self.ims, interval=200, repeat_delay=1000)
            ani.save(f"animation_{self.kernel}.gif", writer='pillow')
            self.ani = False
        return [max_epoch, kernel, C, epsilon, ani, gamma, d, self.accuracy(), np.max(val_accs)]
    
    # accuracy on traning set
    def accuracy(self):
        result = self.predict(self.u)
        return round(np.sum(result == self.z)/self.N, 5)

    def predict(self, x):
        if self.val.shape[0] and self.val_z.shape[0]:
            result = np.sum((self.best_lamd*self.z*self.K_(x).T).T, axis=0) + self.best_w0
        else:
            result = np.sum((self.lambda_*self.z*self.K_(x).T).T, axis=0) + self.w0
        result[result>0] = 1
        result[result<0] = -1
        return result

    def Error(self, idx, num, track):
        if idx == -1:
            return True
        if not track:
            return False
        if idx == 1:
            print(f"[Check Error] Lambda ({num}) is less than zero!!!")
        elif idx == 2:
            print(f"[Check Error] Sum of product of lambda and z ({num}) is not zero!!!")
        elif idx == 3:
            print(f"[Check Error] λi*[Zi(w*ui + w0)-b] != 0 ({num})!!!")
        elif idx == 4:
            print(f"[Check Error] Zi(w*ui + w0)-b < 0 ({num})!!!")
        elif idx == 5:
            print(f"[Check Error] W* != λi*Zi*ui diff={num}!!!")
        return False

    # only checking linearly separable case (no slack) KKT
    def check_KKT_lambda(self, track=False):
        sum_lambda_z = 0
        for i in range(self.N):
            sum_lambda_z += self.lambda_[i]*self.z[i]
            if round(self.lambda_[i], 9) < 0:
                return self.Error(1, self.lambda_[i], track)
        if abs(sum_lambda_z) > 1e-10:
            return self.Error(2, sum_lambda_z, track)
        return self.Error(-1, 0, track)

    # only checking linearly separable case (no slack) KKT
    def check_KKT_w_w0(self, track=False):
#         sum_lambda_z_u = np.full(self.u.shape[1], 0.0)
        for i in range(self.N):
#             check1 = self.lambda_[i]*(self.z[i]*(np.matmul(self.w, self.u[i]) + self.w0)-self.b)
            check1 = self.lambda_[i]*(self.z[i]*self.f_k(i)-self.b)
            if abs(check1) > 1e-10:
                return self.Error(3, check1, track)
            check2 = self.z[i]*self.f_k(i) - self.b
            check2 = 0 if abs(check2) < 1e-10 else check2
            if check2 < 0:
                return self.Error(4, check2, track)
#             sum_lambda_z_u += self.lambda_[i]*self.z[i]*self.u[i]
#         if sum(sum_lambda_z_u != self.w):
#             return self.Error(5, sum_lambda_z_u-self.w, track)
        return self.Error(-1, 0, track)