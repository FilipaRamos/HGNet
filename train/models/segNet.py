from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense

class SegNet():


    def __init__(self, batch_size=32):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=32, strides=1, padding='valid', activation='sigmoid', input_shape=(batch_size, None, 4)))
        self.model.add(Conv2D(16, kernel_size=16, strides=1, padding='valid', activation='sigmoid'))
        self.model.add(Conv2D(8, kernel_size=8, strides=1, padding='valid', activation='relu'))
        self.model.add(Dense(60, input_dim=8, kernel_initializer='normal', activation='relu'))
        self.model.compile()
        
    
        