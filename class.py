
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from cargar_imagenes import cargar_imagenes
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2


class Model():

    def __init__(self):

        self.im_size1 = 256
        self.im_size2 = 256



        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.x_train = np.load('x_train2.npy')
        self.y_train = np.load('y_train2.npy')

        self.y_train_raw = np.array(y_train, np.uint8)
       
        # Normalizar las imagenes
        self.x_train_raw = np.array(x_train, np.float32) / 255.

        print(x_train_raw.shape)
        print(y_train_raw.shape)

        X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw,y_train_raw, test_size=0.2, random_state=42)

        num_class = y_train_raw.shape[1]
        # Transfer Learning



        def transfer_learning:

            base_model = ResNet50(weights = None, include_top=False, input_shape=(im_size1, im_size2, 3))
            return base_model

        def add_top_layer:

            # Add a new top layer
            x = base_model.output
            x = Flatten()(x)
            x = Dropout(0.2)(x)
            x = Dense(32, activation='relu')(x)
            x = Dense(16, activation='relu')(x)
            predictions = Dense(num_class, activation='softmax')(x)

        # This is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # First: train only the top layers (which were randomly initialized)
        #for layer in base_model.layers:
        #    layer.trainable = False

        model.compile(loss='categorical_crossentropy', 
                    optimizer='rmsprop', 
                    metrics=['accuracy'])

        callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', verbose=1)]
        model.summary()

        model.fit(X_train, Y_train, epochs=5, validation_data=(X_valid, Y_valid), verbose=1)

        model.compile(loss='categorical_crossentropy', 
                    optimizer='sgd', 
                    metrics=['accuracy'])

        model.fit(X_train, Y_train, epochs=2, validation_data=(X_valid, Y_valid), verbose=1)

    def load_data()

        x_train = np.load('x_train2.npy')
        y_train = np.load('y_train2.npy')

    