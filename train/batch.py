import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
import keras

""" Custom imports """
import builder

class Batch():
    
    def __init__(self, indexes, fr, batch_size=32, data_path=None, split='train'):
        if data_path is None:
            self.data_path = os.path.join(ROOT_DIR,
                    'kitti/pc_hg_%s.pickle'%(split))
        else:
            self.data_path = data_path
        
        self.fr = fr
        self.indexes = indexes
        self.batch_size = batch_size
        self.batch = 0
        
    def __yield_batch__(self, batch_indexes):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []

        # Generate data
        for batch in batch_indexes:
            grid_, labels_, offsets = self.fr.__getitem__(batch)
            grid = np.expand_dims(grid_, axis=2)
            labels = np.expand_dims(labels_, axis=2)
            #labels = labels_.flatten()
            #print('SAMPLES = ' + str(samples))
            #print('LABELS = ' + str(labels))
            # get samples from builder
            X.append(grid)
            # Store class
            y.append(labels)
            
        return np.array(X), np.array(y)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        if self.batch_size != 0:
            # Generate indexes of the batch
            batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            batch_indexes = self.indexes
        
        # Generate data
        X, y = self.__yield_batch__(batch_indexes)
        
        return X, y
    
    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))
    
    def __start__(self, nb_of_calls_before_reset):
        while True:
            if self.batch == nb_of_calls_before_reset:
                # reset the generator
                self.batch = 0
            else:
                X, y = self.__getitem__(self.batch)
                yield X, y
                self.batch += 1
                
    def __test__(self, steps):
        step = 0
        while step < steps:
            yield self.__getitem__(self.batch)
            self.batch += 1
            step += 1
            
    def __samples__(self, index):
        return self.__yield_batch__([index])
