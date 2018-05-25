import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
import keras

""" Custom imports """
import builder

class Batch(keras.utils.Sequence):
    
    def __init__(self, indexes, fr, num_points=1048, batch_size=32, data_path=None, split='train'):
        if data_path is None:
            self.data_path = os.path.join(ROOT_DIR,
                    'kitti/pc_hg_%s.pickle'%(split))
        else:
            self.data_path = data_path
        
        self.fr = fr
        self.indexes = indexes
        self.batch_size = batch_size
        # using itensity at the same time
        self.n_channels = 4
        self.num_points = num_points
        
    def __yield_batch__(self, batch_indexes):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.num_points, 1, self.n_channels))
        y = np.empty((self.batch_size, self.num_points), dtype=int)

        # Generate data
        for batch, idx in enumerate(batch_indexes):
            samples, labels = self.fr.__getitem__(idx)
            samples = np.expand_dims(samples, axis=1)
            # get samples from builder
            X[batch,] = samples
            # Store class
            y[batch,] = labels
            
        return X, y
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X, y = self.__yield_batch__(batch_indexes)
        
        return X, y
    
    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))
        
        