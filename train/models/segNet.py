from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

class SegNet():


    def __init__(self, batch_size=32, num_points=1048, num_classes=2):
        self.model = Sequential()
        
        self.model.add(Conv2D(64, (1, 1), padding='valid', strides=(1,1), input_shape=(num_points, 1, 4)))
        self.model.add(Conv2D(64, (1, 1), padding='valid', strides=(1,1), activation='relu'))
        self.model.add(Conv2D(64, (1, 1), padding='valid', strides=(1,1), activation='relu'))
        self.model.add(Conv2D(128, (1, 1), padding='valid', strides=(1,1), activation='relu'))
        self.model.add(Conv2D(1024, (1, 1), padding='valid', strides=(1,1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(num_points, 1)))
        
        self.model.add(Conv2D(512, (1, 1), padding='valid', strides=(1,1), activation='relu'))
        self.model.add(Conv2D(256, (1, 1), padding='valid', strides=(1,1), activation='relu'))
        self.model.add(Conv2D(128, (1, 1), padding='valid', strides=(1,1), activation='softmax'))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(64, (1, 1), padding='valid', strides=(1,1), activation='relu'))
        self.model.add(Conv2D(32, (1, 1), padding='valid', strides=(1,1), activation='relu'))
        self.model.add(Conv2D(16, (1, 1), padding='valid', strides=(1,1), activation='softmax'))
        self.model.add(Conv2D(8, (1, 1), padding='valid', strides=(1,1), activation='relu'))
        self.model.add(Conv2D(4, (1, 1), padding='valid', strides=(1,1), activation='softmax'))
        self.model.add(Conv2D(2, (1, 1), padding='valid', strides=(1,1), activation='relu'))
        
        self.model.add(Flatten())
        #self.model.add(Dense(num_points, activation='relu'))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(num_points, activation='relu'))
        self.model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
        
        