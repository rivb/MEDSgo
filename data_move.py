import os
import pandas as pd
import numpy as np
import shutil
import cv2


# abre el archivo csv donde se encuentran las imagenes y categorias
df_train = pd.read_csv('trainLabels.csv')

# revisa todos los indices de trainLabels.csv.
df_train.head()

# tabla con los valores de cada imagen.
targets_series = pd.Series(df_train['level'])

# entrega una tabla con la posicion y el valor de este para cada valor.
one_hot = pd.get_dummies(targets_series, sparse = True)

#https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator

for head in one_hot:

    for pos, source in enumerate(df_train['image']):
        
            img_path =  'resized_train_cropped/resized_train_cropped/'+source+'.jpeg'
            if type(cv2.imread(img_path))==type(None):
                continue

            elif df_train['level'][pos]==head:
                print(head)
                img_final = 'dataset/{}/{}.jpeg'.format(head, source) 

                # Move a file by renaming it's path
                os.rename(img_path, img_final)

            else:

                continue