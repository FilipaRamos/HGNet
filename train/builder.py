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
        grid, labels = hg.height_grid(point_set, box3d=box3d, label=label)
        
        return grid, labels
    
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

def get_ai(fr, model, index):
        ''' Get index-th element from the pickled file dataset. '''
        label = fr.type_list[index]
        point_set = fr.input_list[index]
        if fr.split=='train' or fr.split=='val':
            box3d = fr.box3d_list[index]
        coords_2d = fr.box2d_list[index] # [xmin,ymin,xmax,ymax]
        height = coords_2d[3] - coords_2d[1]
        
        # Get grid offsets
        pc_normalized, X_, offsets, labels = hg.grid_offsets(point_set, height, box3d=box3d, label=label)
        X = np.expand_dims(X_, axis=0)
        X = np.expand_dims(X, axis=3)
        
        logits_ = model.predict(X)
        logits = np.squeeze(logits_)
        
        X_plot = np.squeeze(X_)
        #tools.plot_grid_colormap(X_plot, logits, pred=logits)
        
        # object area to extract connected components
        objA = objArea.objArea(logits, offsets, pc_normalized)
        pc_ = objA.__objCC__()
        
        if pc_ is not None and len(pc_) != 0:
            # get only 2D pc
            pc = pc_.T[:2].T
        else:
            pc = pc_normalized.T[:2].T
        
        pc_norm = augment_pc(pc)
        return pc_norm, labels
    
def create_area_dataset():
    from models import uNet
    from keras.models import load_model
    model_u = load_model('logs/unet.hdf5', custom_objects={'focal_loss_fixed': uNet.focal_loss(gamma=2., alpha=.6), 'f1': uNet.f1})
    print('uNet model is loading...')
    print(model_u.summary())
    
    a_train = Frustum(split='train')
    a_val = Frustum(split='val')
    
    LEN_TRAIN_DATASET = len(a_train)
    LEN_VAL_DATASET = len(a_val)
        
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    
    for i in range(LEN_TRAIN_DATASET):
        out_, labels_ = get_ai(a_train, model_u, i)
        train_data.append(out_)
        train_labels.append(labels_)
        
    for i in range(LEN_VAL_DATASET):
        out_, labels_ = get_ai(a_val, model_u, i)
        val_data.append(out_)
        val_labels.append(labels_)
        
    with open(os.path.join(ROOT_DIR,
                'kitti/pc_ai_train.pickle'),'wb') as ft:
        pickle.dump(train_data, ft)
        pickle.dump(train_labels, ft)
        
    with open(os.path.join(ROOT_DIR,
                'kitti/pc_ai_val.pickle'),'wb') as fv:
        pickle.dump(val_data, fv)
        pickle.dump(val_labels, fv)
                                         
if __name__=='__main__':
    import sys
    from visualization import tools
    from region import objArea
    create_area_dataset()
    """
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
"""