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
import matplotlib.pyplot as plt

class Plot:
    # for complex kernel
#     def plot_kernel(self, x):
#         if self.kernel == 'RBF': # RBF kernel's Φ(x) is infinite dimensional
# #             return np.exp(-self.gamma*np.sum((self.u-x)*(self.u-x), axis=1))
#             # avoid for loop to accelerate computation
#             return np.exp(-self.gamma*((np.sum(self.u*self.u, axis=1) - 2*np.matmul(self.u, x.T).T).T + np.sum(x*x, axis=1)))
#         if self.kernel == 'quadratic':
#             return (np.matmul(self.u, x.T)+1)*(np.matmul(self.u, x.T)+1)
#         return np.matmul(self.u, x.T)
    def K_(self, x):
        return np.matmul(self.u, x.T)
    
    # prediction
    def classify(self, x):
#         result = np.sum(np.multiply(self.lambda_*self.z, np.array([self.plot_kernel(i, x) for i in range(self.N)]).T).T, axis=0) + self.w0
        result = np.sum((self.lambda_*self.z*self.K_(x).T).T, axis=0) + self.w0
        result[result>0], result[result<0]= 0, 1 # class 1 & class 2
        return result
    
    # helper function for plot svm on 2 dim
    def plot(self):
        if len(self.u[0]) > 2:
            print("Cannot draw high dimension!")
            return
        
#         if len(self.w) > 2:
#             self.i0 = np.argmax(self.w)
#             self.i1 = np.argmax(np.delete(self.w, self.i0))
#         plt.clf() # clear previous graph

        class_names = ['Class 1', 'Class 2']

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

        # Labels for each (x1, x2) pair.
        pred_label = self.classify(grid_2d)
        pred_label = np.array(pred_label)

        # reshape the idx (which contains the class label) into an image.
        decision_map = pred_label.reshape(image_size, order='F')

        if not self.ani:
            self.fig = plt.figure(figsize=(6, 6)) # create new
        else:
            if not hasattr(self, 'fig'):
                self.fig = plt.figure(figsize=(6, 6)) # create new
        
        # show the image, give each coordinate a color according to its class
        # label
        im = plt.imshow(decision_map, 
                        vmin=0, 
                        vmax=9,
                        cmap='Pastel1', 
                        extent=[xrange[0], xrange[1], yrange[0], yrange[1]], 
                        origin='lower').findobj()

        # plot the class training data.
        data_point_styles = ['rx', 'bo', 'g*']
        for i in range(2):
            im += plt.plot(self.u[self.z == (1 if not i else -1), 0],
                    self.u[self.z == (1 if not i else -1), 1],
                    data_point_styles[i],
                    label=class_names[i])
        if not self.ani:
            plt.legend()
        
#         # plot support vectors
#         lmd = self.lambda_[self.lambda_>0]
#         for i, (x, y) in enumerate(self.u[self.lambda_>0][:,[0,1]]):
#             if round(lmd[i], 5) > 0:
#                 im += plt.scatter(x, y, s=300, alpha = .5).findobj()

        # only for 2 dim draw
        if self.kernel == 'linear':
            # draw arrow of S1
            bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="r", lw=2)
            degree = np.arctan(self.w[1] / self.w[0])*180/np.pi
            t = plt.text((max_x1*0.3+min_x1*0.7), 
                              (self.w[0]*((max_x1*0.3+min_x1*0.7)) + self.w0)/(-self.w[1]), 
                              "S2" if self.w[0] < 0 else "S1", rotation=degree, size=15, bbox=bbox_props)
            bb = t.get_bbox_patch()
            bb.set_boxstyle("rarrow", pad=0.2)
        
            # draw bound of distance
            b = self.b[0] if type(self.b) != type(1) else self.b # self.b was changed in algebraic method
            # bound 1
            x1 = [min_x1, max_x1]
            y1 = [(self.w[0]*x + self.w0 - b)/(-self.w[1]) for x in x1]
            for i, y in enumerate(y1):
                if y > max_x2: # check up bound
                    x1[i], y1[i] = (b - self.w[1]*max_x2 - self.w0)/self.w[0], max_x2
                if y < min_x2: # check bottom bound
                    x1[i], y1[i] = (b - self.w[1]*min_x2 - self.w0)/self.w[0], min_x2
            plt.plot(x1, y1, 'r.:', label='bound S1', linewidth=1, alpha=0.8)
            # bound 2
            x2 = [min_x1, max_x1]
            y2 = [(self.w[0]*x + self.w0 + b)/(-self.w[1]) for x in x2]
            for i, y in enumerate(y2):
                if y > max_x2: # check up bound
                    x2[i], y2[i] = (-b - self.w[1]*max_x2 - self.w0)/self.w[0], max_x2
                if y < min_x2: # check bottom bound
                    x2[i], y2[i] = (-b - self.w[1]*min_x2 - self.w0)/self.w[0], min_x2
            plt.plot(x2, y2, 'b.:', label='bound S2', linewidth=1, alpha=0.8)

        plt.tight_layout()
        self.ims.append(im)

        return self.fig