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
from .common import *
from .utils import *

class ANN:
    def __init__(self, x, y, model):
        self.x_val = np.array(x)
        self.y_val = np.array(y)
        self.model = model
        
        self.N = len(self.x_val)
        
        self.xdim = 1 if len(self.x_val.shape)==1 else len(self.x_val[0])
        self.ydim = 1 if len(self.y_val.shape)==1 else len(self.y_val[0])
    
    def initialize_w_b(self):
        self.W_val = []
        self.b_val = []
        pre = self.xdim
        for n in self.hidden_layer:
            self.W_val.append(np.random.randn(pre, n))
            self.b_val.append(np.random.randn(n))
            pre = n
    
    def create_network(self):
        self.x_node, self.y_node = Placeholder(), Placeholder()
        self.W_node = [Placeholder() for _ in range(len(self.hidden_layer))]
        self.b_node = [Placeholder() for _ in range(len(self.hidden_layer))]

        self.linear = []
        self.activation = []
        pre = self.x_node
        for i in range(len(self.hidden_layer)):
            self.linear.append(Linear(pre, self.W_node[i], self.b_node[i]))
            if i == len(self.hidden_layer)-1 and self.model == 'regressor':
                break
            self.activation.append(Sigmoid(self.linear[-1]) if self.activation_func == 'Sigmoid' else Relu(self.linear[-1]))
            pre = self.activation[-1]
        self.cost = MSE(self.y_node, self.activation[-1] if self.model=='classifier' else self.linear[-1])
    
    def create_traning_graph(self):
        self.feed_dict = {
            self.x_node: self.x_val,
            self.y_node: self.y_val
        }
        for i in range(len(self.hidden_layer)):
            self.feed_dict[self.W_node[i]] = self.W_val[i]
            self.feed_dict[self.b_node[i]] = self.b_val[i]

        graph = feed_dict_2_graph(self.feed_dict)    # network graph
        self.sorted_graph = topology(graph)          # sorted graph
        self.trainables = self.W_node+self.b_node    # trainable parameters
        
    def fit(self, hidden_layer=[1], activation='Sigmoid', optimizer='SGD', lr=1e-1, epochs=1000, batch_size=16, verbose=True):
        self.hidden_layer = hidden_layer+[self.ydim]
        self.activation_func = activation
        self.optimizer = optimizer
        
        # step 1: initialize weights and bias
        self.initialize_w_b()
        # step 2: Neural network nodes
        self.create_network()
        # step 3: create graph and trainable nodes
        self.create_traning_graph()
        
        # step 4: train
        steps_per_epoch = len(self.x_val) // batch_size
        losses = []

        # for Adam
        it = 0
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        

        for i in range(1, epochs+1):
            loss = 0
            index = list(range(len(self.x_val)))
            np.random.shuffle(index)
            self.x_val = self.x_val[index]
            self.y_val = self.y_val[index]
            for j in range(steps_per_epoch):
                it += 1
                # Step 4.1: Randomly sample a batch of examples and Reset value
                # index = np.random.choice(self.N, batch_size)
                self.x_node.value = self.x_val[j*batch_size:(j+1)*batch_size]
                self.y_node.value = self.y_val[j*batch_size:(j+1)*batch_size]

                # Step 4.2: forward
                for n in self.sorted_graph:
                    n.forward()

                # Step 4.3: backward
                for n in self.sorted_graph[::-1]:
                    n.backward()

                # Step 4.4: optimization
                for t in self.trainables:
                    if self.optimizer == 'SGD':
                        t.value -= lr * t.gradients[t]
                    elif self.optimizer == 'Adam':
                        t.mt[t] = beta1*t.mt[t] + (1-beta1)*t.gradients[t]
                        mt_hat = t.mt[t]/(1-beta1**it)
                        t.vt[t] = beta2*t.vt[t] + (1-beta2)*t.gradients[t]**2
                        vt_hat = t.vt[t]/(1-beta2**it)
                        t.value -= lr * mt_hat / (epsilon + vt_hat**0.5)
                    elif self.optimizer == 'RAdam':
                        ro_inf = 2/(1-beta2) - 1
                        ro_t = ro_inf - 2*it*beta2**it/(1-beta2**it)

                        t.mt[t] = beta1*t.mt[t] + (1-beta1)*t.gradients[t]
                        mt_hat = t.mt[t]/(1-beta1**it)
                        if ro_t <= 4:
                            t.value -= lr * mt_hat
                        else:
                            r_t = np.sqrt((ro_t-4)*(ro_t-2)*ro_inf/((ro_inf-4)*(ro_inf-2)*ro_t))                            
                            t.vt[t] = beta2*t.vt[t] + (1-beta2)*t.gradients[t]**2
                            vt_hat = t.vt[t]/(1-beta2**it)

                            t.value -= lr * mt_hat * r_t / (epsilon + vt_hat**0.5)

                # Step 5: update current loss
                loss += self.sorted_graph[-1].value

            if i % 1 == 0: 
                if verbose:
                    print("Epoch: {}, Loss: {:.4f}".format(i, loss/steps_per_epoch))
                losses.append(loss/steps_per_epoch)
        
        return [hidden_layer, activation, optimizer, lr, epochs, batch_size, losses]
        
    def predict(self, inputs):
        self.x_node.value = inputs
        for n in self.sorted_graph[:-1]:
            n.forward()
        if self.model == 'classifier':
            return np.argmax(self.sorted_graph[-2].value, axis=1)
        return self.sorted_graph[-2].value