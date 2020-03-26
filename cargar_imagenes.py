import cv2
import pandas as pd
from keras.preprocessing import image
import numpy as np
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications.imagenet_utils import preprocess_input

# abre el archivo csv donde se encuentran las imagenes y categorias
df_train = pd.read_csv('trainLabels.csv')

# revisa todos los indices de trainLabels.
df_train.head()

# tabla con los valores de cada imagen.
targets_series = pd.Series(df_train['level'])

# entrega una tabla con la posicion y el valor de este para cada valor.
one_hot = pd.get_dummies(targets_series, sparse = True)

# conversion de los datos en un arreglo de cada posicion con cada valor.
one_hot_labels = np.asarray(one_hot)

img_data_list = []
label_data_list = []

for pos, source in enumerate(df_train['image']):

    img_path =  'resized_train_cropped/resized_train_cropped/'+source+'.jpeg'
    if type(cv2.imread(img_path))==type(None):
        continue

    else:

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_data_list.append(x)

        label = one_hot_labels[pos]
        label_data_list.append(label)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]

num_of_samples = img_data.shape[1]
np.save('x_train2',img_data)
np.save('y_train2',label_data_list)
print('done')