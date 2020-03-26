    
from keras.preprocessing.image import ImageDataGenerator
import keras

import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.layers import Dense, Dropout, Flatten,Input

from tensorflow.python.keras.preprocessing import image


# https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f


train_data_dir = 'dataset'


batch_size =  32
img_height = 224
img_width = 224
nb_epochs = 5
num_class = 5

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation') # set as validation data

base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

x = base_model.output
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)


for layer in base_model.layers:
  layer.trainable = False  # Freeze the layers not to train
  
final_model = keras.models.Model(inputs=base_model.inputs, outputs=predictions) #create final model

final_model = keras.models.Model(inputs=base_model.inputs, outputs=predictions) #create final model


final_model.compile(loss ="categorical_crossentropy", #another term for log loss
                    optimizer = "sgd", 
                    metrics=["accuracy"])


fit_history = final_model.fit_generator(
        train_generator, #train data generator 
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // batch_size,
        epochs = nb_epochs)


# https://brandmark.io/ logo maker

# https://mailchimp.com/ landing page

# share landing page ( facebook , google maps, cold calls, cold mailing)

# business plan 

# user authentication, database, payment, uploads, inference
