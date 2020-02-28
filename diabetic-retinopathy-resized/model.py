
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten

import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2

df_train = pd.read_csv('trainLabels.csv')

df_train.head()


targets_series = pd.Series(df_train['level'])
one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)


im_size1 = 786
im_size2 = 786



x_train = []
y_train = []
x_test = []


i = 0 
for f, breed in tqdm(df_train.values):
    if type(cv2.imread('resized_train_cropped/resized_train_cropped/{}.jpeg'.format(f)))==type(None):
        continue
    else:
        img = cv2.imread('resized_train_cropped/resized_train_cropped/{}.jpeg'.format(f))
        label = one_hot_labels[i]
        x_train.append(cv2.resize(img, (im_size1, im_size2)))
        y_train.append(label)
        i += 1
np.save('x_train2',x_train)
np.save('y_train2',y_train)
print('Done')



x_train = np.load('x_train2.npy')
y_train = np.load('y_train2.npy')



y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.

print(x_train_raw.shape)
print(y_train_raw.shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw,y_train_raw, test_size=0.2, random_state=42)

num_class = y_train_raw.shape[1]

from keras.applications.resnet50 import ResNet50

base_model = ResNet50(weights = None, include_top=False, input_shape=(im_size, im_size, 3))

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