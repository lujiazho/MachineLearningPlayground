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
from .SMoSVC import *

class MultiSVC:
    def __init__(self, z=[], u=[]):
        self.z = z            # classes, 0, 1, 2...
        self.u = u            # data
        self.N = len(z)       # number of points
        
        self.z = np.array(self.z)
        self.u = np.array(self.u)
        
        self.c = len(set(self.z))
        self.svms = []

        self.kernel = ''

        for c in range(self.c):
            cur_z = np.copy(self.z)
            cur_z[cur_z!=c] = -1
            cur_z[cur_z==c] = 1
            cur_u = np.copy(self.u)
            self.svms.append(SMoSVC(z=cur_z, u=cur_u, b=1))
        
    def fit(self, max_epoch=10, kernel='linear', gamma=0.01):
        """
        Args:
            max_epoch: Maximum training epoch if linear KKT cannot be satisfied.
            kernel: Kernel function. Default is linear.
            C: Upper Limit of lambda. A parameter of soft margin/slack-SVM.
            epsilon: Threshold of lambda value change, if which is less than this & KKT is satisfied, then traning ends in advance.
            ani: Launch traning process record as .gif file. This may cause much slower traning because of drawing in each epoch.
            gamma: Parameter of RBF kernel.
            d: Parameter of polynomial kernel.

        Returns:
            Return accuracy on traning set
        """
        self.kernel = kernel
        for c in range(self.c):
            self.svms[c].fit(max_epoch=max_epoch, kernel=self.kernel, gamma=gamma)

        # pred_label = self.predict(self.u)
        return [max_epoch, kernel, gamma] # [round(np.sum(pred_label == self.z)/self.N, 5)]

    def predict(self, x):
        results = []
        for c in range(self.c):
            results.append(np.sum((self.svms[c].lambda_*self.svms[c].z*self.svms[c].K_(x).T).T, axis=0) + self.svms[c].w0)
        results = np.array(results)
        pred_label = np.argmax(results, axis=0)
        return pred_label
    
    def plot(self):
        # Set the feature range for plotting
        max_x1 = np.ceil(np.max(self.u[:, 0])) + 1.0
        min_x1 = np.floor(np.min(self.u[:, 0])) - 1.0
        max_x2 = np.ceil(np.max(self.u[:, 1])) + 1.0
        min_x2 = np.floor(np.min(self.u[:, 1])) - 1.0

        xrange = (min_x1, max_x1)
        yrange = (min_x2, max_x2)

        # step size for how finely you want to visualize the decision boundary.
        inc = int(max_x1 - min_x1)/200

        # generate grid coordinates. This will be the basis of the decision
        # boundary visualization.
        (x1, x2) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc),
                               np.arange(yrange[0], yrange[1] + inc / 100, inc))

        # size of the (x1, x2) image, which will also be the size of the
        # decision boundary image that is used as the plot background.
        image_size = x1.shape
        # make (x1, x2) pairs as a bunch of row vectors.
        grid_2d = np.hstack((x1.reshape(x1.shape[0] * x1.shape[1], 1, order='F'),
                             x2.reshape(x2.shape[0] * x2.shape[1], 1, order='F')))

        pred_label = self.predict(grid_2d)

        # reshape the idx (which contains the class label) into an image.
        decision_map = pred_label.reshape(image_size, order='F')

        self.fig = plt.figure(figsize=(6, 6)) # create new
        
        # show the image, give each coordinate a color according to its class
        # label
        plt.imshow(decision_map, 
                        vmin=0, 
                        vmax=9,
                        cmap='Pastel1', 
                        extent=[xrange[0], xrange[1], yrange[0], yrange[1]], 
                        origin='lower')

        # plot the class training data.
        data_point_styles = ['rx', 'bo', 'g*']
        for i in range(self.c):
            plt.plot(self.u[self.z == i, 0],
                    self.u[self.z == i, 1],
                    data_point_styles[i],
                    label=f'Class {i}')
        plt.legend()
        plt.tight_layout()

        return self.fig