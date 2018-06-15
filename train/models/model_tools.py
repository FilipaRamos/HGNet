import time
import numpy as np
from PIL import Image
import keras.callbacks
import matplotlib.pyplot as plt
from IPython.display import clear_output
from visualization import tools

from models import IoU_tools
        
class WeightsSaver(keras.callbacks.Callback):
    """ 
    This callback class saves weights at each epoch's end
    """
    
    def __init__(self, model, grid_x, grid_y, path='logs/'):
        self.model = model
        self.path = path
        self.grid_x = grid_x
        self.grid_y = grid_y
    
    def on_epoch_end(self, epoch, logs={}):
        name = self.path + 'weights%02d-%03d-%03d.h5' % (epoch, self.grid_x, self.grid_y)
        self.model.save_weights(name)
        
class SaveTrainEx(keras.callbacks.Callback):
    """
    This callback class saves epoch's prediction on images from validation set
    """
    
    def __init__(self, model, indexes, test_gen):
        self.model = model
        self.indexes = indexes
        self.test_gen = test_gen
        
    def on_epoch_end(self, epoch, logs={}):
        for index in self.indexes:
            X, y = self.test_gen.__samples__(index)
            pred = self.model.predict(X)
            X_ = np.squeeze(X)
            y_ = np.squeeze(y)
            pred_ = np.squeeze(pred)
            img = Image.fromarray(pred_)
            img.save('images/' + str(index) + 'pred_' + str(epoch) + '_logits.tiff', 'tiff')
            img_l = Image.fromarray(y_)
            img_l.save('images/' + str(index) + 'label_' + str(epoch) + '_logits.tiff', 'tiff')
            img_x = Image.fromarray(X_)
            img_x.save('images/' + str(index) + 'grid_' + str(epoch) + '_logits.tiff', 'tiff')
        print('Saved some predictions on the test dataset... \n')
            
class PlotLosses(keras.callbacks.Callback):
    """
    This callback class plots the loss/val loss at the end of training
    """
    
    def __init__(self, epoch):
        self.epoch = epoch
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        # plot at each multipleanaconda
        if epoch % self.epoch == 0 and epoch != 0:
            clear_output(wait=True)
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            plt.legend()
            plt.show()
            
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        t = time.time() - self.epoch_time_start
        self.times.append(t)
        print('**' + str(t) + 's **')
            
class IoU(keras.callbacks.Callback):
    """ 
    This callback class calculates iou at each epoch
    TODO: clockwise is not guaranteed so IoU calculation comes out wrong
    """
    
    def __init__(self, model, gen):
        self.model = model
        self.gen = gen
        self.iou = []
    
    def on_epoch_end(self, epoch, logs={}):
        for i in range(len(self.gen)):
            X_, y = self.gen.__getitem__(i)
            X = np.expand_dims(X_, axis=0)
            pred = self.model.predict(X)
            self.iou.append(IoU_tools.iou_bev(y, pred))
        
        print('Mean IoU = ')
        print(np.mean(self.iou))
        self.iou = []