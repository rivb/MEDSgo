import cv2
import pandas as pd
from keras.preprocessing import image
import numpy as np
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.models import Model,Sequential
from keras.layers import Dense, Dropout, Flatten,Input
import keras
import time
# https://github.com/priyanksonis/Automatic-Diabetic-Retinopathy-Detection/blob/master/healthcare/healthcare.py



x_train = np.load('x_train2.npy')
y_train = np.load('y_train2.npy')

img_data = np.array(x_train)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)
num_of_samples = img_data.shape[0]


y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) /255
num_class = y_train_raw.shape[1]

x,y = shuffle(x_train_raw,y_train_raw, random_state= 2)
# Split the dataset
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2)

# https://towardsdatascience.com/how-to-train-your-model-dramatically-faster-9ad063f0f718

img_data=img_data[0]
print(x_train.shape)
print(y_train.shape)

base_model = ResNet50(weights = "imagenet", include_top = False, input_tensor=Input(shape=(224,224,3)))
x = base_model.output
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)

for layer in base_model.layers:
  layer.trainable = False  # Freeze the layers not to train
  
final_model = keras.models.Model(inputs=base_model.inputs, outputs=predictions) #create final model


final_model.compile(loss ="categorical_crossentropy", #another term for log loss
                    optimizer = "adam", 
                    metrics=["accuracy"])


t=time.time()
hist = final_model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=1, validation_data=(x_valid, y_valid))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = final_model.evaluate(x_valid, y_valid, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#custom_resnet_model.save('/home/ee/mtech/eet162639/resnet50_0_30_0.h5')


###########################################################################################################################
###########################################################################################
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
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
print(plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
#fig1.savefig('/home/ee/mtech/eet162639/resnet50_0_30_02.png')