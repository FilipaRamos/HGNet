import numpy as np
from PIL import Image
import keras.callbacks
import matplotlib.pyplot as plt
from IPython.display import clear_output
        
class WeightsSaver(keras.callbacks.Callback):
    """ 
    This callback class saves weights at each epoch's end
    """
    
    def __init__(self, model):
        self.model = model
    
    def on_epoch_end(self, epoch, logs={}):
        name = 'logs/weights%02d.h5' % epoch
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
            pred_ = np.squeeze(pred)
            img = Image.fromarray(pred_)
            img.save('images/' + str(index) + 'pred_' + str(epoch) + '_logits.tiff', 'tiff')
            img_x = Image.fromarray(X_)
            img_x.save('images/' + str(index) + 'grid_' + str(epoch) + '_logits.tiff', 'tiff')
        print('\n Saved some predictions on the test dataset... \n')
            
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
        
        if epoch == self.epoch:
            clear_output(wait=True)
            plt.plot(self.x, self.losses, label="loss")
            plt.plot(self.x, self.val_losses, label="val_loss")
            plt.legend()
            plt.show()