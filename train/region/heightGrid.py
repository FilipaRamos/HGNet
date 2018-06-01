import numpy as np

class HeightGrid():
    
    
    """
        An HeightGrid has a size nxm, a sampling rate, a maximum number of points in the frustum.
    """
    def __init__(self, min_x, max_x, min_y, max_y, min_z, max_z, height_2d, grid_x=160, grid_y=128, intensity=False, label=None):
        x_diff = max_x - min_x
        y_diff = max_y - min_y
        sample_rate_x = x_diff / grid_x
        sample_rate_y = y_diff / grid_y
        
        self.sample_rate = (sample_rate_x, sample_rate_y)
        
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        
        self.n = grid_x
        self.m = grid_y
        
        # height of the object from 2d box (in camera coords)
        self.height_2d = height_2d
        
        self.min_height = min_z
        self.intensity = intensity
        self.labels = np.zeros((self.n, self.m))
        
        self.grid = np.full((self.n, self.m), self.min_height)
        
        if label is not None:
            self.grid_offsets = (label, (self.min_x, self.max_x), (self.min_y, self.max_y), self.sample_rate, (self.n, self.m), self.height_2d)
        else:
            self.grid_offsets = ((self.min_x, self.max_x), (self.min_y, self.max_y), self.sample_rate, (self.n, self.m), self.height_2d)
        
    def __create_grid__(self, points, labels_idx, box3d=None):
        import math
        
        for point, label in zip(points, labels_idx):
            x = int((np.abs(self.min_x) + point[0])/self.sample_rate[0])
            y = int((np.abs(self.min_y) + point[1])/self.sample_rate[1])
            
            # last point contained on the last cell and not starting a new one
            if point[0] == self.max_x:
                x = self.n - 1
            if point[1] == self.max_y:
                y = self.m - 1
            
            if point[2] > self.grid[x][y]:
                # keep intensity and height
                if self.intensity:
                    self.grid[x][y] = (point[2], point[3])
                else:
                    self.grid[x][y] = point[2]
            
            if box3d is not None:
                if label:
                    self.labels[x][y] = 1

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
            max_col_elem(points, 3), min_col_elem(points, 3), \
            max_col_elem(points, 4), min_col_elem(points, 4)
        
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

def sample_points(xmax, xmin, ymax, ymin, zmax, zmin, max_samples):
    x = np.random.uniform(low=xmin, high=xmax, size=max_samples)
    y = np.random.uniform(low=ymin, high=ymax, size=max_samples)
    z = np.random.uniform(low=zmin, high=zmax, size=max_samples)
    intensity = np.random.uniform(low=0, high=1, size=max_samples)
    
    return np.vstack((x,y,z, intensity)).T

def normalize_data(points):
    # first remove intensity 
    points = points.T[0:3].T
    l = points.shape[0]
    centroid = np.mean(points, axis=0)
    pc = points - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def normalize_box3d(box, centroid, m):
    box3d = box - centroid
    box_3d = box3d / m
    return box_3d

def switch_coord_system(pc):
    pc[[0,1,2,3]] = pc[[2,0,1,3]]
    return pc.T

def height_grid(pc, height_2d, box3d=None, label=None):
    """
        Does the whole process, from building the height grid to obtaining the labeled grid
    """
    # get indexes for segmentation first
    if box3d is not None:
        pc_l, labels_idx = extract_pc_in_box3d(pc, box3d)
    else:
        labels_idx = np.zeros((len(pc)))

    # normalize frustum data
    pc_, centroid, m = normalize_data(pc)
    # append intensity again
    pc__ = np.vstack((pc_.T, pc.T[3])).T
    # change to xyz coordinate system
    pc_s = switch_coord_system(pc__.T)
    
    if box3d is not None:
        # normalize 3d box as well
        # only available if training
        box_3d = normalize_box3d(box3d, centroid, m)
    
    """
    if label is not None:
        class_ratio = mean_sizes[label][0] / mean_sizes[label][1]
        grid_x = pc_s.shape[0]
        grid_y = pc_s.shape[0] / class_ratio
    else:
        grid_x = pc_s.shape[0] * 1.2
        grid_y = grid_x / 1.5
    """
    
    xmax, xmin, ymax, ymin, zmax, zmin, intmax, intmin = pc_frame(pc_s)
    hg = HeightGrid(xmin, xmax, ymin, ymax, zmin, zmax, height_2d, intensity=False, label=label)
    hg.__create_grid__(pc_s, labels_idx, box3d=box_3d)
    return hg.grid, hg.labels, hg.grid_offsets

def mean_sizes(pc, box3d, label):
    # get indexes for segmentation first
    pc_l, labels_idx = extract_pc_in_box3d(pc, box3d)

    # normalize frustum data
    pc_, centroid, m = normalize_data(pc)
    # append intensity again
    pc__ = np.vstack((pc_.T, pc.T[3])).T
    # change to xyz coordinate system
    pc_s = switch_coord_system(pc__.T)
    
    # normalize 3d box as well
    # only available if training
    box_3d = normalize_box3d(box3d, centroid, m)
    box_t = np.copy(box_3d).T
    box_t[[0,1,2]] = box_t[[2,0,1]]
    
    return box_t.T