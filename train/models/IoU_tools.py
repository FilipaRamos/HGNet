import numpy as np
from scipy.spatial import ConvexHull
import math

""" These scripts are collected by 
Charles R. Qi
Date: September 2017

Found at https://github.com/charlesq34/frustum-pointnets/blob/master/train/box_util.py on 6th June 2018
"""

def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
     **points have to be counter-clockwise ordered**
    Return:
     a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
    
    def computeIntersection():
        dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
        dp = [ s[0] - e[0], s[1] - e[1] ]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0] 
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
        return(outputList)
    
def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

"""
 Following scripts by me
"""
def box_center(box):
    min_b = np.min(box, axis=0)
    xmin = min_b[0]
    ymin = min_b[1]
    max_b = np.max(box, axis=0)
    xmax = max_b[0]
    ymax = max_b[1]
    
    return np.array([(xmax-xmin) / 2, (ymax - ymin) / 2])  

def clockwiseangle_and_distance(point):
    origin = [0, 0]
    refvec = [0, 1]
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector
    
def clockwise(box):
    return np.array(sorted(list(box), key=clockwiseangle_and_distance))
    
"""
def clockwise_order(box):
    center = box_center(box)
    clockwise = []
    first = []
    second = []
    third = []
    fourth = []
    
    for coord in box:
        if coord[0] < center[0]:
            if coord[1] > center[1]:
                first.append(coord)
            else:
                fourth.append(coord)
        else:
            if coord[1] > center[1]:
                second.append(coord)
            else:
                third.append(coord)
                
    if len(first) > 1:
        if first[0][0] < first[1][0]:
            clockwise.append(first[0])
            clockwise.append(first[1])
        elif first[0][0] > first[1][0]:
            clockwise.append(first[1])
            clockwise.append(first[0])
        else:
            if first[0][1] < first[1][1]:
                clockwise.append(first[0])
                clockwise.append(first[1])
            else:
                clockwise.append(first[1])
                clockwise.append(first[0])
    else:
        if len(first) > 0:
            clockwise.append(first[0])
        
    if len(second) > 1:
        if second[0][0] < second[1][0]:
            clockwise.append(second[0])
            clockwise.append(second[1])
        elif second[0][0] > second[1][0]:
            clockwise.append(second[1])
            clockwise.append(second[0])
        else:
            if second[0][1] < second[1][1]:
                clockwise.append(second[0])
                clockwise.append(second[1])
            else:
                clockwise.append(second[1])
                clockwise.append(second[0])
    else:
        if len(second) > 0:
            clockwise.append(second[0])
        
    if len(third) > 1:
        if third[0][0] < third[1][0]:
            clockwise.append(third[0])
            clockwise.append(third[1])
        elif third[0][0] > third[1][0]:
            clockwise.append(third[1])
            clockwise.append(third[0])
        else:
            if third[0][1] < third[1][1]:
                clockwise.append(third[0])
                clockwise.append(third[1])
            else:
                clockwise.append(third[1])
                clockwise.append(third[0])
    else:
        if len(third) > 0:
            clockwise.append(third[0])
        
    if len(fourth) > 1:
        if fourth[0][0] < fourth[1][0]:
            clockwise.append(fourth[0])
            clockwise.append(fourth[1])
        elif fourth[0][0] > fourth[1][0]:
            clockwise.append(fourth[1])
            clockwise.append(fourth[0])
        else:
            if fourth[0][1] < fourth[1][1]:
                clockwise.append(fourth[0])
                clockwise.append(fourth[1])
            else:
                clockwise.append(fourth[1])
                clockwise.append(fourth[0])
    else:
        if len(fourth) > 0:
            clockwise.append(fourth[0])
        
    return np.array(clockwise)"""
        
def iou_bev(y_true, y_pred):
    y_true = np.reshape(y_true, (4,2))
    y_pred = np.reshape(y_pred, (4,2))
    
    rect1 = np.array([y_true[:,0], y_true[:,1]]).T
    rect2 = np.array([y_pred[:,0], y_pred[:,1]]).T
    area1 = poly_area(np.array(y_true)[:,0], np.array(y_true)[:,1])
    area2 = poly_area(np.array(y_pred)[:,0], np.array(y_pred)[:,1])
    inter, inter_area = convex_hull_intersection(y_true, y_pred)
    
    if (area1+area2-inter_area) == 0:
        iou = 0
    else:
        iou = inter_area/(area1+area2-inter_area)
    
    return iou