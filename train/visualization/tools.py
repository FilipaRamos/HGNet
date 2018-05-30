"""
     Some tools for visualizing pointclouds and images
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_img(image):
    cv2.imshow('camera view',img)
    cv2.waitKey(0)

"""
    Plots the frustum in BEV
"""
def plot_BEV(points, samples=None, labels=None):
    from mpl_toolkits.mplot3d import Axes3D
    params = {"ytick.color" : "w",
              "xtick.color" : "w",
              "axes.labelcolor" : "w",
              "axes.edgecolor" : "w"}
    plt.rcParams.update(params)
    
    fig = plt.figure(figsize=(20,12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((0.11, 0.11, 0.11))
    fig.view_init(90, 180)
    
    pnt = points.T[0:3]
    ax.scatter(*pnt, s = 0.5, c='black', marker='.', alpha=1)
    
    if samples is not None:
        if labels is not None:
            for i, sample in enumerate(samples):
                if labels[i] == 1:
                    ax.plot([sample[0]], [sample[1]], [sample[2]], markerfacecolor='g', markeredgecolor='g', marker='o', markersize=5, alpha=0.6)
                else:
                    ax.plot([sample[0]], [sample[1]], [sample[2]], markerfacecolor='r', markeredgecolor='r', marker='+', markersize=5, alpha=0.6)
        else:
            ax.scatter(*samples.T[0:3], s = 0.5, c='red', marker='.', alpha=1)
        
    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')
    
    #ax.set_xlim([-5, 25])
    #ax.set_ylim([-1, 10])
    
    plt.title('BEV Frustum')

    plt.draw()
    plt.show()
    
def plot_all(img, pc):
    from mpl_toolkits.mplot3d import Axes3D
    params = {"ytick.color" : "w",
              "xtick.color" : "w",
              "axes.labelcolor" : "w",
              "axes.edgecolor" : "w"}
    #plt.rcParams.update(params)
    
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    
    fig = plt.subplot(212, projection='3d')
    fig.set_facecolor((0.11, 0.11, 0.11))
    fig.view_init(90, 180)
    
    pnt = pc.T[0:3]
    fig.scatter(pnt.T[0], pc.T[1], s=0.5, c='black', marker='.', alpha=0.7)
    
    plt.draw()
    plt.tight_layout()
    plt.show()
    
def plot_grid_colormap(grid, labels, pred=None, sig=False):
    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(1,3,1)
    ax.set_title('ColorMap of BEV HeightGrid')
    plt.imshow(grid)
    ax.set_aspect('equal')
    
    ax_l = fig.add_subplot(1,3,2)
    ax_l.set_title('Labels of BEV HeightGrid')
    plt.imshow(labels)
    ax_l.set_aspect('equal')
    
    if pred is not None:
        ax_c = fig.add_subplot(1,3,3)
        ax_c.set_title('Predicted logits of BEV HeightGrid')
        plt.imshow(pred)
        ax_c.set_aspect('equal')
        
        if sig:
            ax_c = fig.add_subplot(1,3,3)
            ax_c.set_title('Predicted logits of BEV HeightGrid')
            plt.imshow(pred)
            ax_c.set_aspect('equal')
        
    plt.colorbar(orientation='horizontal')
    plt.show()
    
def save_img(grid, label, pred, epoch):
    """ Save grid image """
    cmap_grid = plt.cm.jet
    image_grid = cmap_grid(grid)
    # save the image
    plt.imsave('images/grid_' + str(epoch) + '.png', image_grid)
    """ Save original label """
    cmap_label = plt.cm.jet
    image_label = cmap_label(pred)
    # save the image
    plt.imsave('images/label_' + str(epoch) + '.png', image_label)
    """ Save prediction image """
    cmap_pred = plt.cm.jet
    image_pred = cmap_pred(pred)
    # save the image
    plt.imsave('images/pred_' + str(epoch) + '.png', image_pred)