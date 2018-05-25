import os
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

""" Custom imports """
from region import heightGrid as hg

class Frustum():
    
    
    def __init__(self, index, split='train', data_path=None):
        self.index = index
        
        if data_path is None:
            data_path = os.path.join(ROOT_DIR,
                'kitti/pc_hg_%s.pickle'%(split))

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
        cl = self.type_list[index]
        point_set = self.input_list[index]
        #box3d_center = self.get_box3d_center(index)
        #heading_angle = self.heading_list[index]
        
        # Get height grid and turn labels into binary mask
        return hg.height_grid(self.input_list[index], cl, self.box3d_list[index])
                                         
if __name__=='__main__':
    from visualization import tools
    index = 45
    fr = Frustum(index)
    from visualization import vtk_toolkit as vtk
    vtk.plot_pc(points=fr.input_list[index].T[0:3].T)
    samples, labels = fr.__getitem__(index)
    tools.plot_BEV(fr.input_list[index], samples=samples, labels=labels)