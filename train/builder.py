import os
import pickle
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

""" Custom imports """
from region import heightGrid as hg

class Frustum():
    
    
    def __init__(self, split='train', data_path=None):       
        if data_path is None:
            data_path = os.path.join(ROOT_DIR,
                'kitti/pc_hg_%s.pickle'%(split))
            
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
            
        # Get height grid and turn labels into binary mask
        grid, labels, offsets = hg.height_grid(point_set, box3d=box3d, label=label)
        return grid, labels, offsets  
                                         
if __name__=='__main__':
    import sys
    from visualization import tools
    index = int(sys.argv[1])
    fr = Frustum()
    from visualization import vtk_toolkit as vtk
    #vtk.plot_pc(points=fr.input_list[index].T[0:3].T)
    grid_, labels_, offsets = fr.__getitem__(index)
    grid = np.expand_dims(grid_, axis=3)
    labels = np.expand_dims(labels_, axis=3)
    l = np.copy(labels)
    #print(np.count_nonzero(labels))
    #print(grid.shape)
    #tools.plot_grid_colormap(grid, labels)
    equal = 0
    diff = 0
    for batch, batch_pred in zip(labels, l):
        for x, x_pred in zip(batch, batch_pred):
            if x[0] == x_pred[0]:
                equal += 1
            else:
                diff += 1
    print(equal, diff)