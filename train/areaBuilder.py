import os
import keras
import pickle
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

class Area():
    
    def __init__(self, split='train', data_path=None):
        if data_path is None:
            data_path = os.path.join(ROOT_DIR,
                'kitti/pc_ai_%s.pickle'%(split))
            
        self.split = split
        
        with open(data_path,'rb') as fp:
            self.pc_list = pickle.load(fp, encoding='latin1')
            self.label_list = pickle.load(fp, encoding='latin1')
            
    def __len__(self):
        return len(self.pc_list)

    def __getitem__(self, index):
        ''' Get index-th element from the pickled file dataset. '''
        return self.pc_list[index], self.label_list[index].flatten()
