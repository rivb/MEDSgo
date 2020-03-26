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

y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) /255
num_class = y_train_raw.shape[1]
print(type(num_class))
'''
x,y = shuffle(x_train_raw,y_train_raw, random_state= 2)
# Split the dataset
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2)

# https://towardsdatascience.com/how-to-train-your-model-dramatically-faster-9ad063f0f718

print(x_train.shape)
print(y_train.shape)

# https://github.com/fchollet/deep-learning-models/issues/24
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
                    optimizer = "sgd", 
                    metrics=["accuracy"])


t=time.time()
hist = final_model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=1, validation_data=(x_valid, y_valid))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = final_model.evaluate(x_valid, y_valid, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

# custom_resnet_model.save('/home/ee/mtech/eet162639/resnet50_0_30_0.h5')
# https://github.com/arcaduf/nature_paper_predicting_dr_progression/blob/master/create_cnn_pillars/cnn_train.py
# https://www.nature.com/articles/s41598-019-47181-w
# https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized
# https://github.com/priyanksonis/Automatic-Diabetic-Retinopathy-Detection/blob/master/healthcare/healthcare.py
# https://www.youtube.com/watch?v=4HnSc0ppbZk
# https://github.com/mannybernabe/transferLearning_pneumonia/blob/master/Transfer_Learning_Xray_Pneumonia.ipynb


###########################################################################################################################
###########################################################################################
import matplotlib
matplotlib.use('Agg')



import matplotlib.pyplot as plt
#https://stackoverflow.com/questions/42689066/convolutional-neural-net-keras-val-acc-keyerror-acc
#https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''