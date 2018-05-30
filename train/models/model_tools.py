import numpy as np
import keras.callbacks
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(keras.callbacks.Callback):
    
    def __init__(self, model, test_generator):
        self.model = model
        self.gen = test_generator
        
        # metrics to be computed
        self.f1 = []
        self.precision = []
        self.recall = []
        
    def on_epoch_end(self, epoch, logs):
        
        for value in self.gen:
            X, y = value
            batch_pred = self.model.predict(X)
            
            pred = np.round(np.squeeze(batch_pred))
            y_true = np.round(np.squeeze(y))

            _val_f1 = f1_score(y_true, pred, average='macro')
            _val_recall = recall_score(y_true, pred, average='macro')
            _val_precision = precision_score(y_true, pred, average='macro')

            # Micro
            self.f1.append(_val_f1)
            self.precision.append(_val_recall)
            self.recall.append(_val_precision)
                
        print('\n — val_f1: %f — val_precision: %f — val_recall %f' % (np.mean(self.f1), np.mean(self.precision), np.mean(self.recall)))
        
        # empty metrics for the new epoch
        self.f1 = []
        self.precision = []
        self.recall = []
        
class ShowPred(keras.callbacks.Callback):
    
    def __init__(self, model, grid, labels):
        self.model = model
        self.grid = grid
        self.labels = labels

    def on_epoch_end(self, epoch, logs={}):
        result = self.model.predict(self.grid)
        shape_expected = self.grid.shape[0] * self.grid.shape[1] * self.grid.shape[2] * self.grid.shape[3]
        #print('Got ' + str((result == self.labels).sum()) + ' equal labels out of ' + str(shape_expected))
        #print(str(((result == self.labels).sum() / shape_expected) * 100) + ' % accuracy')
        #self.eval_custom_acc(self.labels, result, shape_expected)
        
        from visualization import tools
        grid = np.squeeze(self.grid)
        labels = np.squeeze(self.labels)
        res = np.squeeze(result)
        tools.save_img(grid[10], labels[10], res[10], epoch)
        tools.plot_grid_colormap(grid[10], labels[10], pred=res[10])
        
    def eval_custom_acc(self, labels, pred, shape_expected):
        batch_stats = []
        for batch, batch_pred in zip(labels, pred):
            class0_eq = 0
            num_0 = 0
            class1_eq = 0
            num_1 = 0
            for x, x_pred in zip(batch, batch_pred):
                for y, y_pred in zip(x, x_pred):
                    if y[0] == 0:
                        num_0 += 1
                    else:
                        num_1 += 1
                    if y[0] == y_pred[0]:
                        if y[0] == 0:
                            class0_eq += 1
                        else:
                            class1_eq += 1
            batch_stats.append([class0_eq, num_0, class1_eq, num_1])
        bs = np.array(batch_stats)
        mean = np.mean(bs, axis=0)
        print('Class 0: ' + str(mean[0]) + ' / ' + str(mean[1]) + ' -> acc = ' + str(mean[0]/mean[1]))
        print('Class 1: ' + str(mean[2]) + ' / ' + str(mean[3]) + ' -> acc = ' + str(mean[2]/mean[3]))
        
class WeightsSaver(keras.callbacks.Callback):
    
    def __init__(self, model):
        self.model = model
    
    def on_epoch_end(self, epoch, logs={}):
        name = 'logs/weights%02d.h5' % epoch
        self.model.save_weights(name)