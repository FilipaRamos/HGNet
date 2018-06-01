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
        #grid, labels, offsets = hg.height_grid(point_set, box3d=box3d, label=label)
        box = hg.mean_sizes(point_set, box3d, label)
        s = box[box[:,2].argsort()]
        bev = s[4:]
        
        return label, bev, point_set
                                         
if __name__=='__main__':
    import sys
    from visualization import tools
    fr = Frustum(split='train')
    size = int(len(fr))
    car_sizes = []
    ped_sizes = []
    cyc_size = []
    for i in range(size):
        label, bev, point_set = fr.__getitem__(i)
        y = bev[bev[:,1].argsort()]
        x = bev[bev[:,0].argsort()]
        x_size = np.abs(x[2][0] - x[0][0])
        y_size = np.abs(y[2][1] - y[0][1])
        if x_size > y_size:
            width = y_size
            depth = x_size
        else:
            width = x_size
            depth = y_size
        if label == 'Car':
            car_sizes.append([width, depth])
        elif label == 'Pedestrian':
            ped_sizes.append([width, depth])
        elif label == 'Cyclist':
            cyc_size.append([width, depth])
    
    print('Car mean sizes')
    print(np.mean(car_sizes, axis=0))
    print('MAX w,d')
    print(max([size[0] for size in car_sizes]), max([size[1] for size in car_sizes]))
    
    print('Ped mean sizes')
    print(np.mean(ped_sizes, axis=0))
    print('MAX w,d')
    print(max([size[0] for size in ped_sizes]), max([size[1] for size in ped_sizes]))
    
    print('Cyc mean sizes')
    print(np.mean(cyc_size, axis=0))
    print('MAX w,d')
    print(max([size[0] for size in cyc_size]), max([size[1] for size in cyc_size]))

