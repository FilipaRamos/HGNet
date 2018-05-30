from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Lambda, ZeroPadding2D, BatchNormalization, Activation, UpSampling2D, Layer, Reshape, Permute
from keras.optimizers import SGD, Adam
from keras import backend as K
import tensorflow as tf

class SegNet():


    def __init__(self, batch_size=32, num_classes=2, num_channels=1, init_weights=False):
        
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=(2,2), padding='valid', strides=(1,1), input_shape=(100, 50, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool'))
        self.model.add(Flatten(name='flat'))
        self.model.add(Dense(5000, activation='sigmoid', name='out'))
        
        sgd = SGD(lr=0.00001)
        self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
        #show_output(self.model)
        
        """
        kernel = 3
        filter_size = 64
        pad = 1
        pool_size = 2

        self.model = Sequential()
        self.model.add(Layer(input_shape=(160, 96, 1)))

        # encoder
        self.model.add(ZeroPadding2D(padding=(pad,pad)))
        self.model.add(Conv2D(filter_size, (kernel, kernel), padding='valid'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        self.model.add(ZeroPadding2D(padding=(pad,pad)))
        self.model.add(Conv2D(128, (kernel, kernel), padding='valid'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        self.model.add(ZeroPadding2D(padding=(pad,pad)))
        self.model.add(Conv2D(256, (kernel, kernel), padding='valid'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        self.model.add(ZeroPadding2D(padding=(pad,pad)))
        self.model.add(Conv2D(512, (kernel, kernel), padding='valid'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))


        # decoder
        self.model.add( ZeroPadding2D(padding=(pad,pad)))
        self.model.add( Conv2D(512, (kernel, kernel), padding='valid'))
        self.model.add( BatchNormalization())

        self.model.add( UpSampling2D(size=(pool_size,pool_size)))
        self.model.add( ZeroPadding2D(padding=(pad,pad)))
        self.model.add( Conv2D(256, (kernel, kernel), padding='valid'))	
        self.model.add( BatchNormalization())

        self.model.add( UpSampling2D(size=(pool_size,pool_size)))
        self.model.add( ZeroPadding2D(padding=(pad,pad)))
        self.model.add( Conv2D(128, (kernel, kernel), padding='valid'))
        self.model.add( BatchNormalization())

        self.model.add( UpSampling2D(size=(pool_size,pool_size)))
        print(self.model.output_shape)
        self.model.add( ZeroPadding2D(padding=(pad,pad)))
        #$self.model.add(ZeroPadding2D(((1, 0), (0, 0))))
        self.model.add( Conv2D(filter_size, (kernel, kernel), padding='valid'))
        self.model.add( BatchNormalization())


        self.model.add(Conv2D(2, (1, 1), padding='valid'))

        self.model.outputHeight = self.model.output_shape[-2]
        self.model.outputWidth = self.model.output_shape[-1]
        print(self.model.output_shape)

        self.model.add(Reshape((2, self.model.output_shape[-2]*self.model.output_shape[-1]), input_shape=(2, self.model.output_shape[-2], self.model.output_shape[-1])))
        print(self.model.output_shape)
        self.model.add(Permute((2, 1)))
        self.model.add(Activation('softmax'))
        sgd = SGD(lr=0.00001)
        self.model.compile(loss="categorical_crossentropy", optimizer= sgd , metrics=['accuracy'] )
        """
        
def gen_weights(shape1, shape2):
    import numpy as np
    weights = []
    for idx in range(shape1):
        row_unique = np.random.rand()
        row = np.full((shape2), row_unique)
        weights.append(row)
    return [np.array(weights), np.zeros((shape2))]

def show_w(model):
    for layer in model.layers:
        g=layer.get_config()
        h=layer.get_weights()
        print (g)
        print (h)
        print(h[0].shape)
        print(h[1].shape)
        
def show_output(model):
    for layer in model.layers:
        cfg = layer.get_config()
        output = layer.output
        print(cfg)
        print(output)
    
        