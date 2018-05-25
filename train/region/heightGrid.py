import math
import numpy as np

mean_sizes = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
            'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
            'Cyclist': np.array([1.76282397,0.59706367,1.73698127])}

class HeightGrid():
    
    
    """
        An HeightGrid has a size nxm, a sampling rate, a maximum number of points in the frustum.
    """
    def __init__(self, label, min_z, max_z, min_x, max_x, max_y, sample_rate_z=0.01, sample_rate_x=0.005):
        self.label = label
        self.sample_rate = (sample_rate_z, sample_rate_x)
        
        self.min_z = min_z
        self.max_z = max_z
        self.min_x = min_x
        self.max_x = max_x
        
        self.max_height = max_y
        
        #self.n = len(np.arange(self.min_z, self.max_z, self.sample_rate[0])) - 1
        #self.m = len(np.arange(self.min_x, self.max_x, self.sample_rate[1])) - 1
        
        self.samples = []
        
    def __create_grid__(self, points):
        steps_z = np.arange(self.min_z, self.max_z, self.sample_rate[0])
        steps_x = np.arange(self.min_x, self.max_x, self.sample_rate[1])
        
        grid = self.__fill_dict__(steps_z, steps_x)
        
        for point in points:
            z = int((-self.min_z + point[2])/self.sample_rate[0])
            x = int((-self.min_x + point[0])/self.sample_rate[1])
            
            cell = str(steps_z[z]) + '-' + str(steps_x[x])
            if point[1] < grid.get(cell)[1]: #and not math.isclose(point[2], 0.0, rel_tol=1e-02, abs_tol=1e-01):
                val = grid.get(cell)
                grid[cell] = [val[0], point[1], val[2], point[3]]
            
        # dict values are the sampled points
        #print(grid.values())
        self.samples = [point for point in list(grid.values()) if point[1] != self.max_height]
    
    def __fill_dict__(self, steps_z, steps_x):
        grid = {}
        
        for z in steps_z:
            for x in steps_x:
                cell = str(z) + '-' + str(x)
                grid[cell] = [x + (self.sample_rate[1]/2), self.max_height, z + (self.sample_rate[0]/2), 0]
        
        return grid

def max_col_elem(l, col):
    return max(l, key=lambda x: x[col - 1])[col - 1]

def min_col_elem(l, col):
    return min(l, key=lambda x: x[col - 1])[col - 1]

"""
    Maximum and minimum points in the pointcloud in all dimensions (z,x,y)
"""
def pc_frame(points):
    return max_col_elem(points, 1), min_col_elem(points, 1), \
            max_col_elem(points, 2), min_col_elem(points, 2), \
            max_col_elem(points, 3), min_col_elem(points, 3)
        
def samples_to_array(hg):
    return np.array([np.array(point) for point in hg.samples])
        
def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0
        
def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds
        
def map_labels(hg, box3d):
    points, indexes = extract_pc_in_box3d(samples_to_array(hg), box3d)
    return [int(index) for index in indexes]

def sample_points(xmax, xmin, ymax, ymin, zmax, zmin, max_samples):
    x = np.random.uniform(low=xmin, high=xmax, size=max_samples)
    y = np.random.uniform(low=ymin, high=ymax, size=max_samples)
    z = np.random.uniform(low=zmin, high=zmax, size=max_samples)
    intensity = np.random.uniform(low=0, high=1, size=max_samples)
    
    return np.vstack((x,y,z, intensity)).T
        

def height_grid(pc, label, box3d):
    """
        Does the whole process, from frustum to building the height grid and obtaining the samples
    """
    xmax, xmin, ymax, ymin, zmax, zmin = pc_frame(pc)
    if label == 'Car':
        hg = HeightGrid(label, zmin, zmax, xmin, xmax, ymax, sample_rate_z=0.10, sample_rate_x=0.07)
    elif label == 'Pedestrian':
        hg = HeightGrid(label, zmin, zmax, xmin, xmax, ymax, sample_rate_z=0.01, sample_rate_x=0.05)
    elif label == 'Cyclist':
        hg = HeightGrid(label, zmin, zmax, xmin, xmax, ymax, sample_rate_z=0.03, sample_rate_x=0.05)
    hg.__create_grid__(pc)
    return hg.samples, map_labels(hg, box3d)