# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 04:33:26 2019

@author: baris_akin98
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing import image as image_utils


file_path="C:/Users/User/Desktop/Deep Learning/train"
normal="/NORMAL"
pnomoni="/PNEUMONIA"

x=[]
y=[]

def img_to_array(image_file_path):
    image=image_utils.load_img(image_file_path,target_size=(32,32))
    image=image_utils.img_to_array(image)
    image=image/255.0
    return image 


for classes in os.listdir(file_path):
    if classes=="NORMAL":
        label_of_normal=0
        for i in os.listdir(file_path+normal):
            array_of_image=img_to_array(file_path+normal+"/"+i)
            x.append(array_of_image)
            y.append(label_of_normal)
        
    elif classes=="PNEUMONIA":
        label_of_pneumonia=1
        for i in os.listdir(file_path+pnomoni):
            array_of_image=img_to_array(file_path+pnomoni+"/"+i)
            x.append(array_of_image)
            y.append(label_of_pneumonia)
            
x=np.asarray(x)
y=np.asarray(y)

from keras.utils.np_utils import  to_categorical
y=to_categorical(y,num_classes=2)

x_val=[]
y_val=[]
file_path2="C:/Users/User/Desktop/Deep Learning/val"
for classes in os.listdir(file_path2):
    if classes=="NORMAL":
        label_of_normal=0
        for i in os.listdir(file_path2+normal):
            array_of_image=img_to_array(file_path2+normal+"/"+i)
            x_val.append(array_of_image)
            y_val.append(label_of_normal)
        
    elif classes=="PNEUMONIA":
        label_of_pneumonia=1
        for i in os.listdir(file_path2+pnomoni):
            array_of_image=img_to_array(file_path2+pnomoni+"/"+i)
            x_val.append(array_of_image)
            y_val.append(label_of_pneumonia)
            
x_val=np.asarray(x_val)
y_val=np.asarray(y_val)
y_val=to_categorical(y_val,num_classes=2)

from keras.models import Sequential
from keras.layers import Dropout,Dense,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

model=Sequential()

model.add(Conv2D(filters=30,kernel_size=(20,20),padding="same",activation="relu",input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.40))

model.add(Conv2D(filters=20,kernel_size=(15,15),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.30))

model.add(Conv2D(filters=10,kernel_size=(5,5),padding="valid",activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(64,activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(4,activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(2,activation="softmax"))
optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
epochs=20
batch_size=20
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
datagen.fit(x)
history = model.fit_generator(datagen.flow(x,y, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val), steps_per_epoch=x.shape[0] // batch_size)



file_path3="C:/Users/User/Desktop/Deep Learning/Veri Setleri/chest_xray/test"
normal="/NORMAL"
pnomoni="/PNEUMONIA"

x_test=[]
y_test=[]

for classes in os.listdir(file_path3):
    if classes=="NORMAL":
        label_of_normal=0
        for i in os.listdir(file_path3+normal):
            array_of_image=img_to_array(file_path3+normal+"/"+i)
            x_test.append(array_of_image)
            y_test.append(label_of_normal)
        
    elif classes=="PNEUMONIA":
        label_of_pneumonia=1
        for i in os.listdir(file_path3+pnomoni):
            array_of_image=img_to_array(file_path3+pnomoni+"/"+i)
            x_test.append(array_of_image)
            y_test.append(label_of_pneumonia)
            
x_test=np.asarray(x_test)
y_test=np.asarray(y_test)

y_test=to_categorical(y_test,num_classes=2)

model.evaluate(x_test,y_test,batch_size=batch_size)
























           