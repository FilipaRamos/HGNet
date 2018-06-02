import math
import numpy as np
from scipy.ndimage.measurements import label

max_sizes = {'Car': np.array([2.1654689554006055,5.909382149376355]),
            'Pedestrian': np.array([0.895747106198418,1.085311911692555]),
            'Cyclist': np.array([0.259199703380381,0.6659377439736036])}

grid_x=160
grid_y=128

class objArea():
    
    def __init__(self, logits, offsets, pc):
        self.logits = logits
        self.offsets = offsets
        self.pc = pc
        
        self.n_comp = 0
        
        self.rnd = []
        self.components_grid = []
        self.c_list = []
        
        self.component = []
        
    def __filter_seg__(self, tresh=0.5):
        """
            Finds the pixel indexes above a certain tresh
        """
        indexes_z = self.logits <= tresh
        indexes_o = self.logits > tresh
        self.rnd = np.copy(self.logits)
        self.rnd[indexes_z] = 0
        self.rnd[indexes_o] = 1
              
    def __cc__(self):
        structure = np.ones((3, 3), dtype=np.int)
        self.components_grid, self.n_comp = label(self.rnd, structure)
        
        print('Found ' + str(self.n_comp) + ' distinct component.')
        print('2D detection believes there is a ' + str(self.offsets[0]) + ' on the frustum.')
        
        indexes = np.indices(self.rnd.shape).T[:,:,[1, 0]]
        for i in range(1, self.n_comp+1):
            self.c_list.append(self.components_grid == i)
            
    # only the component with largest confidence logits is kept
    def __component_confidence__(self):
        confidence = 0
        index = 0
        idx = 0
        for c in self.c_list:
            area = self.logits[c]
            score = area.mean()
            if score > confidence:
                confidence = score
                index = idx
            idx += 1
        self.component = self.c_list[index]
            
    # gets the hg sample rates from the offsets
    def __pixel_value__(self):
        x_diff = self.offsets[1][1] - self.offsets[1][0]
        y_diff = self.offsets[2][1] - self.offsets[2][0]
        sample_rate_x = x_diff / grid_x
        sample_rate_y = y_diff / grid_y
        
        return sample_rate_x, sample_rate_y
    
    def __pixel_conv__(self, min_v, pixel_idx, sample_rate):
        return min_v + pixel_idx * sample_rate
    
    def __pixel_to_lidar__(self, pixel_area, sample_rate_x, sample_rate_y):
        (min_x_pixel, max_x_pixel), (min_y_pixel, max_y_pixel) = pixel_area
        
        min_x = self.__pixel_conv__(self.offsets[1][0], min_x_pixel, sample_rate_x)
        max_x = self.__pixel_conv__(self.offsets[1][0], max_x_pixel+1, sample_rate_x)
        
        min_y = self.__pixel_conv__(self.offsets[2][0], min_y_pixel, sample_rate_y)
        max_y = self.__pixel_conv__(self.offsets[2][0], max_y_pixel+1, sample_rate_y)
        
        # return in hull slice function format
        return (min_x, min_y, max_x, max_y)
            
    def __build_areas__(self, pv):
        # sample rates
        pv_x, pv_y = pv
        # get area corresponding to class
        area_m_max_size = max(max_sizes[self.offsets[0]])
        area_pixel_x = math.ceil(area_m_max_size/pv_x)
        area_pixel_y = math.ceil(area_m_max_size/pv_y)
        
        # pixel boundaries on area (pixel coordinates)
        min_pixel_x = min([idx[0] for idx in self.component])
        max_pixel_x = max([idx[0] for idx in self.component])
        min_pixel_y = min([idx[1] for idx in self.component])
        max_pixel_y = max([idx[1] for idx in self.component])

        pixel_area = (min_pixel_x, max_pixel_x + area_pixel_x), (min_pixel_y - area_pixel_y, max_pixel_y + area_pixel_y)
        # turn it back to lidar coordinates in bev
        bev_lidar_box = self.__pixel_to_lidar__(pixel_area, pv_x, pv_y)
        pc_, indexes = extract_pc_in_box2d(self.pc, bev_lidar_box)
         
        return pc_
            
    # function that does everything, from slicing area of interest to returning pc inside the area of interest
    def __objCC__(self, tresh=0.5):
        self.__filter_seg__(tresh=tresh)
        self.__cc__()
        self.component = self.c_list[0]
        if self.n_comp > 1:
            self.__component_confidence__()
        pv = self.__pixel_value__()
        
        # returns areas with the coordinates
        return self.__build_areas__(pv)
    
def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]] 
    box2d_corners[1,:] = [box2d[2],box2d[1]] 
    box2d_corners[2,:] = [box2d[2],box2d[3]] 
    box2d_corners[3,:] = [box2d[0],box2d[3]] 
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:], box2d_roi_inds