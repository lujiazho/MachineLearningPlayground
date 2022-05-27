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
from matplotlib.pyplot import cm

def self_cdist(nclass, xy, sample_mean):
    dim = len(xy[0])
    dis = []
    for i in range(nclass):
        class_ = (np.hstack([xy for _ in range(nclass)]) - sample_mean.reshape(1, -1))[:,i*dim:(i+1)*dim]
        dis.append(np.linalg.norm(class_, axis=1).reshape(-1,1))
    return np.hstack(dis)

def plotDecBoundByMean(training, label_train, sample_mean, title=""):
    nclass =  len(np.unique(label_train))

    # Set the feature range for ploting
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    inc = 0.005

    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), 
                         np.arange(yrange[0], yrange[1]+inc/100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack((x.reshape(x.shape[0]*x.shape[1], 1, order='F'), 
                    y.reshape(y.shape[0]*y.shape[1], 1, order='F'))) # make (x,y) pairs as a bunch of row vectors.

    # alternative: dist_mat = scipy.spatial.distance.cdist(xy, sample_mean)
    dist_mat = self_cdist(nclass, xy, sample_mean)
    
    pred_label = np.argmin(dist_mat, axis=1)

    decisionmap = pred_label.reshape(image_size, order='F')

    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

    # plot the class training data.
    plt.plot(training[label_train == 0, 0],training[label_train == 0, 1], 'rx')
    plt.plot(training[label_train == 1, 0],training[label_train == 1, 1], 'go')
    if nclass == 3:
        plt.plot(training[label_train == 2, 0],training[label_train == 2, 1], 'b*')

    # include legend for training data
    if nclass == 3:
        l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
    else:
        l = plt.legend(('Class 1', 'Class 2'), loc=2)
    plt.gca().add_artist(l)

    # plot the class mean vector.
    m1, = plt.plot(sample_mean[0,0], sample_mean[0,1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
    m2, = plt.plot(sample_mean[1,0], sample_mean[1,1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
    if nclass == 3:
        m3, = plt.plot(sample_mean[2,0], sample_mean[2,1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')

    # include legend for class mean vector
    if nclass == 3:
        l1 = plt.legend([m1,m2,m3],['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)
    else:
        l1 = plt.legend([m1,m2], ['Class 1 Mean', 'Class 2 Mean'], loc=4)

    plt.gca().add_artist(l1)
    plt.title(title)
    plt.show()

def plotKMean(training, k, sample_mean, title=""):
    plt.figure(figsize=(8,8))

    # Set the feature range for ploting
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    inc = 0.005

    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) # make (x,y) pairs as a bunch of row vectors.

    # alternative: dist_mat = scipy.spatial.distance.cdist(xy, sample_mean)
    dist_mat = self_cdist(k, xy, sample_mean)
    
    pred_label = np.argmin(dist_mat, axis=1)

    decisionmap = pred_label.reshape(image_size, order='F')

    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower', cmap=plt.cm.Paired)

    # plot the class training data.
    plt.plot(training[:, 0], training[:, 1], 'g.', markersize=2)

    # plot the class mean vector.
    plt.plot(sample_mean[:,0], sample_mean[:,1], 'x', markersize=24, color='w')

    plt.title(title)
    plt.show()


def plotBoundByDecisionFunction(training, label_train, decision_function, inc=0.1, augmented=False):
    nclass = len(np.unique(label_train))

    x1_idx, x2_idx = 0, 1
    # for criterion function: g(x) = w*x, which w and x are augmented, 1 in x is at 0th place
    if augmented:
        x1_idx, x2_idx = 1, 2

    # Set the feature range for plotting
    max_x1 = np.ceil(np.max(training[:, x1_idx])) + 1.0
    min_x1 = np.floor(np.min(training[:, x1_idx])) - 1.0
    max_x2 = np.ceil(np.max(training[:, x2_idx])) + 1.0
    min_x2 = np.floor(np.min(training[:, x2_idx])) - 1.0

    xrange = (min_x1, max_x1)
    yrange = (min_x2, max_x2)

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
    pred_label = decision_function(grid_2d)

    decision_map = pred_label.reshape(image_size, order='F')

    plt.imshow(decision_map, cmap=plt.cm.Paired,
              extent=[xrange[0], xrange[1], yrange[0], yrange[1]],
              origin='lower')

    # plot the class training data.
    class_names = ['Class ' + str(int(c+1)) for c in range(nclass)]
    data_point_styles = ['rx', 'bo', 'g*']
    color = iter(cm.rainbow(np.linspace(0, 1, nclass)))
    for i in range(nclass):
        plt.scatter(training[label_train == i, x1_idx],
                training[label_train == i, x2_idx],
                color=next(color),
                label=class_names[i])
    plt.legend()

    plt.tight_layout()
    plt.show()