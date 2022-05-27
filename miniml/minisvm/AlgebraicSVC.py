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
from .Plot import *

class AlgebraicSVC(Plot):
    def __init__(self, z=[], u=[], b=1):
        self.z = np.array(z)  # classes, only for two: 1 or -1
        self.u = np.array(u)  # data
        self.N = len(z)       # number of points
        
        # parameters for calc lambda
        self.A = []           # A in Aρ=b
        self.b = b            # margin, which is also one value for b in Aρ=b
        self.lambda_ = []     # lambda
        self.miu = None       # miu
        
        # parameters for calc w
        self.w = []           # optimal w
        self.w0 = None        # optimal bias term w0
        
        # fixed, do not change
        self.ani = False
        self.ims = []
        self.kernel = 'linear'
        
    # calculate A and b in Aρ=b
    def cal_A_b(self):
        if type(self.z) == list and type(self.u) == list:
            return None, None
        if type(self.A) != list and type(self.b) != list:
            return self.A, self.b
        # calculate A
        A = []
        for i in range(self.N):
            Ai = []
            for j in range(self.N):
                term = self.z[j]*self.z[i]*np.matmul(self.u[j], self.u[i])
                Ai.append(term)
            Ai.append(-self.z[i])
            A.append(Ai)
        A.append([self.z[i] for i in range(self.N)] + [0])
        self.A = np.array(A)
        # calculate b
        self.b = np.full(self.N+1, self.b)
        self.b[-1] = 0 # this is for sum of z_i*lambda_i = 0
        return self.A, self.b
    
    # Use NumPy to invert matrix, and to calculate lambda and µ
    def calc_lambda_miu(self):
        if self.miu != None:
            return self.lambda_, self.miu
        A, b = self.cal_A_b()
        if type(A) == type(None) or type(b) == type(None):
            return None, None
        # calculate ρ in Aρ=b
        ρ = np.matmul(np.linalg.inv(A), b)
        self.lambda_, self.miu = ρ[:-1], ρ[-1]
        return self.lambda_, self.miu
    
    # assign certain lambda to zero
    def calc_lambda_miu_with_zero_lambda(self, zeros=[]):
        # after deleting, A becomes NxN, b becomes Nx1
        A, b = np.delete(self.A, zeros, axis=1), np.delete(self.b, zeros) # delete the columns in list of zeros
        A = np.delete(A, zeros, axis=0) # delete the rows in list of zeros
        if type(A) == type(None) or type(b) == type(None):
            return None, None
        # calculate ρ in Aρ=b
        ρ = np.matmul(np.linalg.inv(A), b)
        self.lambda_, self.miu = ρ[:-1], ρ[-1]
        zeros.sort()
        for loc in zeros:
            self.lambda_ = np.insert(self.lambda_, loc, 0) # insert the fixed zero lambda
        return self.lambda_, self.miu
    
    # Calculate the optimal (nonaugmented) weight vector w∗ and w0
    def calc_w_w0(self):
        # calculate for optimal w
        self.w = np.full(2, 0.0)
        for i in range(self.N):
            self.w += self.lambda_[i]*self.z[i]*self.u[i]
        # calculate for optimal w0
        for i in range(self.N):
            if self.lambda_[i] != 0:
                self.w0 = self.b[0]/self.z[i] - np.matmul(self.w, self.u[i])
                break
        return self.w, self.w0
    
    # Check lambda satisfies the KKT conditions
    def check_KKT_lambda(self):
        sum_lambda_z = 0
        for i in range(self.N):
            sum_lambda_z += self.lambda_[i]*self.z[i]
            if self.lambda_[i] < 0:
                print(f"[Check Error] Lambda {self.lambda_[i]} is less than zero!!!")
                return False
        if abs(sum_lambda_z) > 1e-10:
            print(f"[Check Error] Sum of product of lambda and z ({sum_lambda_z}) is not zero!!!")
            return False
        return True

    # Check that the resulting w and w0 satisfy the KKT conditions
    def check_KKT_w_w0(self):
        sum_lambda_z_u = np.full(2, 0.0)
        for i in range(self.N):
            check1 = self.lambda_[i]*(self.z[i]*(np.matmul(self.w, self.u[i]) + self.w0)-self.b[0])
            if abs(check1) > 1e-10:
                print(f"[Check Error] λi*[Zi(w*ui + w0)-1] != 0 ({check1})!!!")
                return False
            check2 = self.z[i]*(np.matmul(self.w, self.u[i]) + self.w0) - self.b[0]
            check2 = 0 if abs(check2) < 1e-10 else check2
            if check2 < 0:
                print(f"[Check Error] Zi(w*ui + w0)-1 < 0 ({check2})!!!")
                return False
            sum_lambda_z_u += self.lambda_[i]*self.z[i]*self.u[i]
        if sum(sum_lambda_z_u != self.w):
            print(f"[Check Error] W* != λi*Zi*ui {sum_lambda_z_u} != {self.w}!!!")
            return False
        return True