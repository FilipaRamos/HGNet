import os
import keras
import pickle
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

""" Custom imports """
from region import objArea
from region import heightGrid as hg

from visualization import tools

class Area():
    
    def __init__(self, model, split='train', data_path=None):
        if data_path is None:
            data_path = os.path.join(ROOT_DIR,
                'kitti/pc_hg_%s.pickle'%(split))
            
        self.model = model
        self.split = split
        
        with open(data_path,'rb') as fp:
            self.id_list = pickle.load(fp, encoding='latin1')
            self.box2d_list = pickle.load(fp, encoding='latin1')
            self.bev_list = pickle.load(fp, encoding='latin1')
            self.box3d_list = pickle.load(fp, encoding='latin1')
            self.input_list = pickle.load(fp, encoding='latin1')
            self.label_list = pickle.load(fp, encoding='latin1')
            self.type_list = pickle.load(fp, encoding='latin1')
            self.heading_list = pickle.load(fp, encoding='latin1')
            self.size_list = pickle.load(fp, encoding='latin1')
            # frustum_angle is clockwise angle from positive x-axis
            self.frustum_angle_list = pickle.load(fp, encoding='latin1')
            
    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        ''' Get index-th element from the pickled file dataset. '''
        label = self.type_list[index]
        point_set = self.input_list[index]
        if self.split=='train' or self.split=='val':
            box3d = self.box3d_list[index]
        coords_2d = self.box2d_list[index] # [xmin,ymin,xmax,ymax]
        height = coords_2d[3] - coords_2d[1]
        
        # Get grid offsets
        pc_normalized, X_, offsets, labels = hg.grid_offsets(point_set, height, box3d=box3d, label=label)
        X = np.expand_dims(X_, axis=0)
        X = np.expand_dims(X, axis=3)
        
        logits_ = self.model.predict(X)
        logits = np.squeeze(logits_)
        
        X_plot = np.squeeze(X_)
        tools.plot_grid_colormap(X_plot, logits, pred=logits)
        
        # object area to extract connected components
        objA = objArea.objArea(logits, offsets, pc_normalized)
        pc_ = objA.__objCC__()
        
        # get only 2D pc
        pc = pc_.T[:2].T
        pc_norm = augment_pc(pc)
        
        return pc_norm, labels
    
def max_col_elem(l, col):
    return max(l, key=lambda x: x[col - 1])[col - 1]

def min_col_elem(l, col):
    return min(l, key=lambda x: x[col - 1])[col - 1]
    
def samples_to_array(pc):
    return np.array([np.array(xy) for xy in pc])
    
def sample_points(xmax, xmin, ymax, ymin, max_samples):
    x = np.random.uniform(low=xmin, high=xmax, size=max_samples)
    y = np.random.uniform(low=ymin, high=ymax, size=max_samples)
    
    return np.array([x, y]).T
    
def unsample_pc(pc, nr):
    for n in range(nr):
        random_index = np.random.randint(0, len(pc))
        pc.pop(random_index)
    return pc
    
def augment_pc(pc, size=128):
    samples = []
    pc_ = list(pc)
    if len(pc_) < size:
        nr_samples = size - len(pc_)
        samples = sample_points(max_col_elem(pc_, 1), min_col_elem(pc_, 1), max_col_elem(pc_, 2), min_col_elem(pc_, 2), nr_samples)
        pc_ = pc_ + list(samples)
    elif len(pc_) > size:
        nr_deletions = len(pc_) - size
        pc_ = unsample_pc(pc_, nr_deletions)
    
    return samples_to_array(pc_)
        
    