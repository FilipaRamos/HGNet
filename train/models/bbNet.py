from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Lambda, ZeroPadding2D, BatchNormalization, Activation, UpSampling2D, Layer, Reshape, Permute, Input, merge, Dense
from keras.optimizers import SGD, Adam
import keras.backend as K
import tensorflow as tf
K.set_image_dim_ordering('th')

import numpy as np
import math

""" Custom imports for IoU calculations """
from models import IoU_tools

class bbNet():
    
    def __init__(self, nr_points=128, channels=2):
        # input points are (1, N, 2)
        input_ = Input(shape=(nr_points, channels))
        print(input_.shape)
        conv1_ = Conv1D(64, 4, activation='relu', padding='same')(input_)
        print(conv1_.shape)
        conv1_b = BatchNormalization()(conv1_)
        conv1_d = Dropout(0.5)(conv1_b)
        conv2_ = Conv1D(128, 4, activation='relu', padding='same')(conv1_d)
        print(conv2_.shape)
        m_pool = MaxPooling1D()(conv2_)
        print(m_pool.shape)
        flat = Flatten()(m_pool)
        dense1_ = Dense(64, activation='relu')(flat)
        dense2_ = Dense(8, activation='softmax')(dense1_)
        print(dense1_.shape)
        print(dense2_.shape)
        self.model = Model(inputs=input_, outputs=dense2_)
        self.model.compile(optimizer=Adam(lr = 1e-4), loss=['mean_squared_error'], metrics=['accuracy'])

        
def euclidean_dist(x1, x2, y1, y2):
    return math.hypot(x2 - x1, y2 - y1)

def euclidean_array(y_true, y_pred):
    return np.array([euclidean_dist(p1[0], p2[0], p1[1], p2[1]) for p1 in y_true for p2 in y_pred])
    
def corner_loss(y_true, y_pred):
    """
        y_true and y_pred: (4,2)
        anti clockwise from first quadrant 
    """
    y_true = K.reshape(y_true, [-1, 2])
    y_pred = K.reshape(y_true, [-1, 2])
    
    #min_t = K.min(y_true, axis=0)
    #max_t = K.max(y_true, axis=0)
    #min_p = K.min(y_pred, axis=0)
    #max_p = K.max(y_pred, axis=0)
    
    #return K.mean(K.square(y_true - y_pred)) + K.square(np.subtract(max_t, min_t)/2 - np.subtract(max_p, min_p)/2)
    return K.mean(K.square(y_pred - y_true))

def corner_acc(y_true, y_pred):
    """
        y_true and y_pred: (4,2)
        anti clockwise from first quadrant 
    """
    y_true = K.reshape(y_true, [-1, 2])
    y_pred = K.reshape(y_true, [-1, 2])
    
    min_t = K.min(y_true, axis=0)
    max_t = K.max(y_true, axis=0)
    min_p = K.min(y_pred, axis=0)
    max_p = K.max(y_pred, axis=0)
    
    return 2*((K.mean(K.abs(y_true - y_pred)) + K.abs(np.subtract(max_t, min_t)/2 - np.subtract(max_p, min_p)/2)) / (K.mean(K.abs(y_true - y_pred)) + K.abs(np.subtract(max_t, min_t)/2 - np.subtract(max_p, min_p)/2) + K.epsilon()))