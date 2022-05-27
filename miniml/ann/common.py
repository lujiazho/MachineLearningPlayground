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

# parameter node
class Placeholder(Node):
    def __init__(self):
        Node.__init__(self)
         # For Adam optimizer
        self.mt = {self:0}
        self.vt = {self:0}

    def forward(self, value=None):
        if value is not None:
            self.value = value
        
    def backward(self):
        self.gradients = {self:0}
        

        for n in self.outputs:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

# functional node: add all inputs' values
class Add(Node):
    def __init__(self, *nodes):
        Node.__init__(self, nodes)

    def forward(self):
        self.value = sum(map(lambda n: n.value, self.inputs))

# functional node: calculate linear function
class Linear(Node):
    def __init__(self, nodes, weights, bias):
        Node.__init__(self, [nodes, weights, bias])

    def forward(self):
        inputs = self.inputs[0].value
        weights = self.inputs[1].value
        bias = self.inputs[2].value

        self.value = np.dot(inputs, weights) + bias
        
    def backward(self):
        # create zero loss for all inputs
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        for n in self.outputs:
            # Get the cost w.r.t this node.
            grad_cost = n.gradients[self]

            self.gradients[self.inputs[0]] += np.dot(grad_cost, self.inputs[1].value.T)
            self.gradients[self.inputs[1]] += np.dot(self.inputs[0].value.T, grad_cost)
            self.gradients[self.inputs[2]] += np.sum(grad_cost, axis=0, keepdims=False)

# functional node: calculate sigmoid function
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1./(1 + np.exp(-1 * x))

    def forward(self):
        self.x = self.inputs[0].value
        self.value = self._sigmoid(self.x)

    def backward(self):
        self.partial = self._sigmoid(self.x) * (1 - self._sigmoid(self.x))

        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

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
        self.gradients = {n: np.zeros_like(n.value) for n in self.inputs}

        for n in self.outputs:
            grad_cost = n.gradients[self]

            self.gradients[self.inputs[0]] += grad_cost * self._partial(self.x)

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
