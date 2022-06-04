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

# primary node
class Node:
    def __init__(self, inputs=[]):
        self.inputs = inputs
        self.outputs = []

        for n in self.inputs:
            n.outputs.append(self)

        self.value = None
        self.gradients = {}

    def forward(self):
        raise NotImplemented
    
    def backward(self):
        raise NotImplemented

    def optimize(self):
        raise NotImplemented

# parameter node
class Placeholder(Node):
    def __init__(self):
        Node.__init__(self)
         # For Adam optimizer
        self.mt = 0
        self.vt = 0

    def forward(self, value=None):
        if value is not None:
            self.value = value
        
    def backward(self):
        self.gradients = {self:0}

        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

    def optimize(self, optimizer='SGD', lr=0.001, it=0):
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        if optimizer == 'SGD':
            self.value -= lr * self.gradients[self]
        elif optimizer == 'Adam':
            self.mt = beta1*self.mt + (1-beta1)*self.gradients[self]
            mt_hat = self.mt/(1-beta1**it)
            self.vt = beta2*self.vt + (1-beta2)*self.gradients[self]**2
            vt_hat = self.vt/(1-beta2**it)
            self.value -= lr * mt_hat / (epsilon + vt_hat**0.5)
        elif optimizer == 'RAdam':
            ro_inf = 2/(1-beta2) - 1
            ro_t = ro_inf - 2*it*beta2**it/(1-beta2**it)

            self.mt = beta1*self.mt + (1-beta1)*self.gradients[self]
            mt_hat = self.mt/(1-beta1**it)
            if ro_t <= 4:
                self.value -= lr * mt_hat
            else:
                r_t = np.sqrt((ro_t-4)*(ro_t-2)*ro_inf/((ro_inf-4)*(ro_inf-2)*ro_t))                            
                self.vt = beta2*self.vt + (1-beta2)*self.gradients[self]**2
                vt_hat = self.vt/(1-beta2**it)

                self.value -= lr * mt_hat * r_t / (epsilon + vt_hat**0.5)

# functional node: add all inputs' values
class Add(Node):
    def __init__(self, *nodes):
        Node.__init__(self, list(nodes))

    def forward(self):
        self.value = sum(map(lambda n: n.value, self.inputs))

    def backward(self):
        # create zero loss for all inputs
        self.gradients = {n: np.zeros_like(n.value, dtype=np.float64) for n in self.inputs}

        for n in self.outputs:
            # Get the cost w.r.t this node.
            grad_cost = n.gradients[self]

            for i in range(len(self.inputs)-1):
                self.gradients[self.inputs[i]] += grad_cost
            self.gradients[self.inputs[-1]] += np.sum(grad_cost, axis=0, keepdims=False)

# functional node: calculate linear function
class Linear(Node):
    def __init__(self, nodes, weights, bias):
        self.bias = True if bias else False
        Node.__init__(self, [nodes, weights, bias] if bias else [nodes, weights])

    def forward(self):
        inputs = self.inputs[0].value
        weights = self.inputs[1].value
        if self.bias:
            bias = self.inputs[2].value

        self.value = np.dot(inputs, weights) + (bias if self.bias else 0)
        
    def backward(self):
        # create zero loss for all inputs
        self.gradients = {n: np.zeros_like(n.value, dtype=np.float64) for n in self.inputs}

        for n in self.outputs:
            # Get the cost w.r.t this node.
            grad_cost = n.gradients[self]

            self.gradients[self.inputs[0]] += np.dot(grad_cost, self.inputs[1].value.T)
            self.gradients[self.inputs[1]] += np.dot(self.inputs[0].value.T, grad_cost)
            if self.bias:
                self.gradients[self.inputs[2]] += np.sum(grad_cost, axis=0, keepdims=False)

# simple embedding for nlp
class Embedding(Node):
    def __init__(self, node, embedding):
        Node.__init__(self, [node, embedding])

    def forward(self):
        x = self.inputs[0].value
        embedding = self.inputs[1].value

        self.value = embedding[x].reshape(len(x), -1)
        
    def backward(self):
        self.gradients = {n: np.zeros_like(n.value, dtype=np.float64) for n in self.inputs}
        x = self.inputs[0].value # n*c
        embedding = self.inputs[1].value # w*d

        x_ = np.zeros((x.size, embedding.shape[0]))
        x_[np.arange(x.size), x.flatten()] = 1

        for n in self.outputs:
            grad_cost = n.gradients[self]

            grad_R = grad_cost.reshape((x.shape[0]*x.shape[1], embedding.shape[1]))
            self.gradients[self.inputs[1]] += np.dot(x_.T, grad_R)

            grad_x_ = np.dot(grad_R, embedding.T)
            self.gradients[self.inputs[0]] += np.sum(grad_x_, axis=1, keepdims=False).reshape((x.shape[0], x.shape[1]))

# functional node: calculate sigmoid function
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        res = np.copy(x)
        res[x>0] = 1./(1 + np.exp(-(x[x>0])))
        res[x<0] = np.exp(x[x<0]) / (1.0 + np.exp(x[x<0]))
        return res

    def forward(self):
        self.x = self.inputs[0].value
        self.value = self._sigmoid(self.x)

    def backward(self):
        self.partial = self._sigmoid(self.x) * (1 - self._sigmoid(self.x))
        self.gradients = {n: np.zeros_like(n.value, dtype=np.float64) for n in self.inputs}

        for n in self.outputs:
            grad_cost = n.gradients[self]

            self.gradients[self.inputs[0]] += grad_cost * self.partial

# functional node: calculate relu function
class Relu(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _relu(self, x):
        x[x<0] = 0
        return x

    def _partial(self, x):
        x[x<0] = 0
        x[x>0] = 1
        return x

    def forward(self):
        self.x = self.inputs[0].value
        self.value = self._relu(self.x)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value, dtype=np.float64) for n in self.inputs}

        for n in self.outputs:
            grad_cost = n.gradients[self]

            self.gradients[self.inputs[0]] += grad_cost * self._partial(self.x)

# functional node: tanh function
class Tanh(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _tanh(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def forward(self):
        self.x = self.inputs[0].value
        self.value = self._tanh(self.x)

    def backward(self):
        self.partial = (1 - self._tanh(self.x)**2)
        self.gradients = {n: np.zeros_like(n.value, dtype=np.float64) for n in self.inputs}

        for n in self.outputs:
            grad_cost = n.gradients[self]

            self.gradients[self.inputs[0]] += grad_cost * self.partial

# functional node: calculate MSE function
class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inputs[0].value
        a = self.inputs[1].value
        assert(y.shape == a.shape)

        self.m = self.inputs[0].value.shape[0]
        self.diff = y - a

        self.value = np.mean(self.diff**2)

    # Derivative * loss = Δw = gradients
    def backward(self):
        self.gradients[self.inputs[0]] = (2 / self.m) * self.diff
        self.gradients[self.inputs[1]] = (-2 / self.m) * self.diff
