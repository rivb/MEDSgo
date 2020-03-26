
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2

im_size1 = 224
im_size2 = 224



x_train = []
y_train = []
x_test = []

def cargar_imagenes(num, im_size1, im_size2):

    i = 0 
    for f, breed in tqdm(df_train.values[:num]):
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

img = cv2.imread('resized_train_cropped/resized_train_cropped/15_left.jpeg')
img = cv2.resize(img, (224, 224))
print(img.shape)

'''
cargar_imagenes(1000,im_size1,im_size2)
'''

x_train = np.load('x_train2.npy')
y_train = np.load('y_train2.npy')
y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255

x_train, x_valid, y_train, y_valid = train_test_split(x_train_raw,y_train_raw, test_size=0.2, random_state=42)

num_class = y_train_raw.shape[1]
print(num_class)
'''
from keras.applications.resnet50 import ResNet50


# https://towardsdatascience.com/how-to-train-your-model-dramatically-faster-9ad063f0f718

original_model    = ResNet50(include_top=True,weights='imagenet')
bottleneck_input  = original_model.get_layer(index=0).input
bottleneck_output = original_model.get_layer(index=-2).output
bottleneck_model  = Model(inputs=bottleneck_input, outputs=bottleneck_output)

# shutdown training of ResNet50
for layer in bottleneck_model.layers:
    layer.trainable = False


new_model = Sequential()
new_model.add(bottleneck_model)
x = new_model.output
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)

new_model = Model(inputs=bottleneck_model.input, outputs=predictions)

new_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
hist = new_model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=1, validation_data=(x_valid, y_valid))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = new_model.evaluate(x_valid, y_valid, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

'''
'''
out = Dense(num_class, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable




import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(2)

fig=plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
#fig.savefig('/home/ee/mtech/eet162639/resnet50_0_30_01.png')


fig1=plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
#fig1.savefig('/home/ee/mtech/eet162639/resnet50_0_30_02.png')
'''