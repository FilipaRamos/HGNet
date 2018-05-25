import os
import sys
import argparse
import numpy as np
from PIL import Image
import pickle as pickle

""" Custom packages """
import kitti_utils as utils
import kitti_object as kitti

""" Input """
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

""" PATH bases """
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.join('E:\\', 'fimam', 'Documents', 'KITTI', 'object')

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]] 
    box2d_corners[1,:] = [box2d[2],box2d[1]] 
    box2d_corners[2,:] = [box2d[2],box2d[3]] 
    box2d_corners[3,:] = [box2d[0],box2d[3]] 
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:], box2d_roi_inds

def normalize_data(points):
    l = points.shape[0]
    centroid = np.mean(points, axis=0)
    pc = points - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def extract_data(idx_filename, split, output_filename, type_whitelist=['Car', 'Pedestrian', 'Cyclist']):
    ''' Extract front view pointclouds on rectified coordinates normalized
        
    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti.Object(ROOT_DIR, split=split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = [] # int number
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    bev_box_list = [] # (4, 3) array in rect camera coord, normalized
    box3d_list = [] # (8,3) array in rect camera coord, normalized
    label_list = [] # 1 for roi object, 0 for clutter
    type_list = [] # string e.g. Car
    heading_list = [] # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = [] # array of l,w,h
    frustum_angle_list = [] # array of 2d box center from pos x-axis
    input_list = [] # pointcloud in frustum in rect coords, normalized

    pos_cnt = 0
    all_cnt = 0
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = kitti.get_lidar_in_image_fov(pc_velo[:,0:3],
            calib, 0, 0, img_width, img_height, True)
        
        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist :continue

            # 2D BOX: Get pts rect backprojected 
            box2d = objects[obj_idx].box2d
            
            xmin,ymin,xmax,ymax = box2d
            box_fov_inds = (pc_image_coord[:,0]<xmax) & \
                (pc_image_coord[:,0]>=xmin) & \
                (pc_image_coord[:,1]<ymax) & \
                (pc_image_coord[:,1]>=ymin)
            box_fov_inds = box_fov_inds & img_fov_inds
            pc_in_box_fov = pc_rect[box_fov_inds,:]
            # Get frustum angle (according to center pixel in 2D BOX)
            box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
            uvdepth = np.zeros((1,3))
            uvdepth[0,0:2] = box2d_center
            uvdepth[0,2] = 20 # some random depth
            box2d_center_rect = calib.project_image_to_rect(uvdepth)
            frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
                box2d_center_rect[0,0])
            # 3D BOX: Get pts velo in 3d box
            obj = objects[obj_idx]
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) 
            _,inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
            label = np.zeros((pc_in_box_fov.shape[0]))
            label[inds] = 1
            # Get 3D BOX heading
            heading_angle = obj.ry
            # Get 3D BOX size
            box3d_size = np.array([obj.l, obj.w, obj.h])
            # Get BEV coords after normalizing box
            bev = kitti.bev_box(box3d_pts_3d)

            # Reject too far away object or object without points
            if ymax-ymin<25 or np.sum(label)==0:
                continue
                
            id_list.append(data_idx)
            box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
            bev_box_list.append(bev)
            box3d_list.append(box3d_pts_3d)
            input_list.append(pc_in_box_fov)
            label_list.append(label)
            type_list.append(objects[obj_idx].type)
            heading_list.append(heading_angle)
            box3d_size_list.append(box3d_size)
            frustum_angle_list.append(frustum_angle)

            # collect statistics
            pos_cnt += np.sum(label)
            all_cnt += pc_in_box_fov.shape[0]
        
    print('Average pos ratio: %f' % (pos_cnt/float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt)/len(id_list)))
    
    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(bev_box_list, fp)
        pickle.dump(box3d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_train', action='store_true', help='Generate front view pointcloud from all samples.')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split front view pointcloud.')
    parser.add_argument('--car_only', action='store_true', help='Only generate cars; otherwise cars, pedestrians and cyclists')
    args = parser.parse_args()

    if args.car_only:
        whitelist = ['Car']
        output = 'pc_car_hg_'
    else:
        whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output = 'pc_hg_'

    if args.gen_train:
        extract_data(\
            os.path.join(BASE_DIR, 'sets/train.txt'),
            'training',
            os.path.join(BASE_DIR, output+'train.pickle'),
            type_whitelist=whitelist)

    if args.gen_val:
        extract_data(\
            os.path.join(BASE_DIR, 'sets/val.txt'),
            'training',
            os.path.join(BASE_DIR, output+'val.pickle'),
            type_whitelist=whitelist)
        
