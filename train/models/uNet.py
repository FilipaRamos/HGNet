from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Lambda, ZeroPadding2D, BatchNormalization, Activation, UpSampling2D, Layer, Reshape, Permute, Input, merge
from keras.optimizers import SGD, Adam
import keras.backend as K
import tensorflow as tf

class uNet():
    
    def __init__(self, img_rows, img_cols, batch_size=4, init_weights=False):
        
        out_ch=1 
        start_ch=64
        depth=4
        inc_rate=2.
        activation='relu'
        dropout=0.5
        batchnorm=False
        maxpool=True
        upconv=True
        residual=False
        
        i = Input(shape=(img_rows, img_cols, 1))
        o = self.level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
        o = Conv2D(out_ch, 1, activation='sigmoid')(o)
        self.model = Model(inputs=i, outputs=o)
        self.model.compile(optimizer=Adam(lr = 1e-4), loss=[focal_loss(gamma=2., alpha=.7)], metrics=[f1])
        
    def conv_block(self, m, dim, acti, bn, res, do=0):
        n = Conv2D(dim, 3, activation=acti, padding='same')(m)
        n = BatchNormalization()(n) if bn else n
        n = Dropout(do)(n) if do else n
        n = Conv2D(dim, 3, activation=acti, padding='same')(n)
        n = BatchNormalization()(n) if bn else n
        return Concatenate()([m, n]) if res else n
    
    def level_block(self, m, dim, depth, inc, acti, do, bn, mp, up, res):
        if depth > 0:
            n = self.conv_block(m, dim, acti, bn, res)
            m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
            m = self.level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
            if up:
                m = UpSampling2D()(m)
                m = Conv2D(dim, 2, activation=acti, padding='same')(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
                n = Concatenate()([n, m])
                m = self.conv_block(n, dim, acti, bn, res)
        else:
            m = self.conv_block(m, dim, acti, bn, res, do)
        return m
    
def focal_loss(gamma=2., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):#with tensorflow
        eps = 1e-12
        y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
