import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

""" Custom imports """
import builder

class Batch():
    
    def __init__(self, indexes, batch_size=32, data_path=None, split='train'):
        if data_path is None:
            self.data_path = os.path.join(ROOT_DIR,
                    'kitti/pc_hg_%s.pickle'%(split))
        else:
            self.data_path = data_path
        
        self.start_idx = indexes[0]
        self.end_idx = indexes[1]
        # using itensity at the same time
        self.n_channels = 4
        
    def __yeld_batch__(self):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, None, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for idx in range(self.start_idx, self.end_idx):
            fr = builder.Frustum(idx)
            samples, labels = fr.__getitem__(index)
            
            # get samples from builder
            X[idx,] = samples
            # Store class
            y[idx] = labels

        return X, y
        
        