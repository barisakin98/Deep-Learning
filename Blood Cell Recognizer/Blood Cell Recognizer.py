# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 02:59:45 2019

@author: NeuroPanda
"""
import pandas as pd
import numpy as np
from keras.preprocessing import image as image_utils
import matplotlib.pyplot as plt
import os

file_path="C:/Users/User/Desktop/Deep Learning/dataset2-master/images/TRAIN"
print(os.listdir("C:/Users/User/Desktop/Deep Learning/dataset2-master/images/TRAIN"))

#İmage PreProcessing 
def img_show(img_path):
    img_path="C:/Users/User/Desktop/Deep Learning/dataset-master/JPEGImages/BloodImage_00000.jpg"
    image=image_utils.load_img(img_path)
    image=image_utils.img_to_array(image)
    image=image/255.0  #255 değerleri arasına inmesini sağladık. Bütün bunları tüm resimler için yapacağız.
    plt.imshow(image)
    plt.show()

def image_to_array(img_path):
    image=image_utils.load_img(img_path)
    image=image_utils.img_to_array(image)
    image=image/255.0
    return image

x=[]
y=[]

for img_files_dic in os.listdir(file_path):
    if img_files_dic==os.listdir(file_path)[0]:
        for img_files in os.listdir(file_path+"/"+"EOSINOPHIL"):
            #size_of_train_dataset+=1
            label_of_eosinophil=0
            image_array_values=image_to_array(file_path+"/"+"EOSINOPHIL"+"/"+img_files)
            y.append(label_of_eosinophil)
            x.append(image_array_values)
    elif img_files_dic==os.listdir(file_path)[1]:
        for img_files in os.listdir(file_path+"/"+"LYMPHOCYTE"):
            #size_of_train_dataset+=1
            label_of_lymphocyte=1
            image_array_values=image_to_array(file_path+"/"+"LYMPHOCYTE"+"/"+img_files)
            y.append(label_of_lymphocyte)
            x.append(image_array_values)
    elif img_files_dic==os.listdir(file_path)[2]:
        for img_files in os.listdir(file_path+"/"+"MONOCYTE"):
            #size_of_train_dataset+=1
            label_of_monocyte=2
            image_array_values=image_to_array(file_path+"/"+"MONOCYTE"+"/"+img_files)
            y.append(label_of_monocyte)
            x.append(image_array_values)
    elif img_files_dic in os.listdir(file_path)[3]:
        for img_files in os.listdir(file_path+"/"+"NEUTROPHIL"):
            label_of_neutrophil=3
            image_array_values=image_to_array(file_path+"/"+"NEUTROPHIL"+"/"+img_files)
            y.append(label_of_neutrophil)
            #size_of_train_dataset+=1
            x.append(image_array_values)

x_train=np.asarray(x)
y_train=np.asarray(y)
#y_train=y_train.reshape(-1,1)
#y_train=y_train.T

from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train,num_classes=4)

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1,random_state=3)


from keras.models import Sequential
from keras.layers import Dropout,Dense,Flatten,Conv2D,Maxpool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator


def convolutional_NN_model(x_train,y_train,x_val,y_val):
    
    model=Sequential()
    #Convolutional Layers and Pooling 
    model.add(Conv2D(filters=30,kernel_size=(10,10),padding="Same",activation="relu",input_shape=(240,320,3)))
    model.add(Maxpool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters=25,kernel_size=(10,10),padding="valid",activation="relu"))
    model.add(Maxpool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters=20,kernel_size=(10,10),padding="Same",activation="relu"))
    model.add(Maxpool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Dropout(0.25))
    
    #Flatten process
    model.add(Flatten())
    model.add(Dense(200,activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(150,activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(100,activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(4,activation="softmax"))
    optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
    model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
    epochs=100
    batch_size=100 

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    
    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val), steps_per_epoch=x_train.shape[0] // batch_size)
    return history

convolutional_NN_model(x_train,y_train,x_val,y_val)