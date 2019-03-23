# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 21:36:28 2019

@author: NeuroPanda
"""
import numpy as np
import random
"""np.random.seed(100)
x=np.random.random(50)
np.random.seed(250)
y=np.random.random(50)
np.random.seed(100)
z=np.random.random(50) #dediğimizde  x ile aynı değerde bir matris elde etmiş oluruz."""

#Resmin sayısal karşılıkları pixeller ile ifade edilmektedir.

import pandas as pd
import matplotlib.pyplot as plt
train=pd.read_csv("train.csv") 
#print(train.head()) #data setimizi incelediğimizde (42000 e 784 lük bir yapı. ) [0. index labeli geri kalan 784. index ise toplam pixel sayısını ifade ediyor.]
#[yani fotoğraflarımız 28*28 pixelden meydana geliyor. ]
#ve train data setimizde toplamda 42000 adet veri var. 
test=pd.read_csv("test.csv")#test data setimiz ise 28000 datadan meydana geliyor.

y_train=train["label"]
x_train=train.drop(labels="label",axis=1)

"""img=x_train.iloc[0,:].values  #values komutu önemlidir.! values ile dataframe şeklinde olan verimizi array haline çevirdik.aynısını lineer regresyon dersinde de yapmıştık hatırlarsak.
img=img.reshape((28,28))
plt.imshow(img,cmap="gray")
plt.show()"""


"""NORMALİZATİON 
#Normalizasyon yaptığımızda farklı renklerden kaynaklı hataların önüne geçmiş oluruz. 
#Normalizasyon,tüm renk değerlerini 0 ile 1 arasına taşıma işlemidir. yani siyah-beyaz tonları arasında incelemektir.
"""


"""RESHAPE
Resimlerimiz 28*28.Fakat bunları 3d matrix haline getirmemiz gereklidir. 28*28*1 şeklinde 3d matrix vermemiz 
gereklidir.Eğer renklerimiz grayscale olmasaydı 28*28*3 olarak kullanacaktık."""

"""LABEL ENCODİNG 
TEST setimiz için labellerimizi hazırlayacağız."""

#Normalization: bir değeri 0 ile 1 değeri arasına sıkıştırmaya normalization denir.
x_train=x_train/255.0   #buradaki değerleri 255 e böldük. 
test=test/255.0

#Reshaping data
x_train=x_train.values.reshape(-1,28,28,1) #42000,28,28,1 haline geldi.önceki şekli 42000,784 idi. Son sayıların yani pixel sayılarının uyumlu olmasına dikkat edelim.
test=test.values.reshape(-1,28,28,1) 
#Label Encoding

from keras.utils.np_utils import to_categorical 
y_train=to_categorical(y_train,num_classes=10)


from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1,random_state=2)

"""CONVOLUTION OPERATION
3*3 lük bir filter alalım(Feature Detector). Bu filtreyi resmimize uyguladığımızda feature mapsler elde etmiş oluruz.
Bu filtremiz konveks şekilleri ve düz yerleri tespit etmemizi,filtrelememizi sağlar. 
STRİDE İŞLEMİ,filtremizin 1 er 1 er atlayarak matrixlerin üzerini dolaşması işlemidir. 
Filtre sayımız 1 den fazla olacaktır tabiiki. Farklı filtreler farklı bölgelerin tanınmalarını sağlarlar.
Bu sayede birden fazla featuremap elde etmiş oluruz.
Çizgileri yakalamak için edge detection filtremizi kullanırız.Bu siyah zemin üzerindeki kıvrımlar,edgeler filtreler ile detect edilmektedir.
Tespit etmek istediğimiz şeye göre bu filtrelerle oynamamız gereklidir. 


Convolution işlemi uygulayıp feature mapslerimizi elde ettikten sonra reLu kullanılarak non-linearity artılır.
Convolution işlemi information Loss a neden olmuştu.


PADDİNG İŞLEMİ bu information lossu önlememizi sağlar.bilgiyi tutmamızı sağlayan işlemdir. SAME PADDİNG bu padding yöntemlerinden biridir. Farklı Padding yöntemleri vardır.
"""

"""POOLING OPERATION
Çeşitli pooling yöntemleri vardır. Bunlardan en çok kullanılan max-pooling şlemidir.Farklı Pooling yöntemleri de mevcuttur.
MAX POOLİNG İŞLEMİ,girdimizin ve işlem yaptığımız size çok büyük olursa yavaş çalışır. 
Max pooling işlemi down-sampling yapar. hacmi küçültür. ayrıca over-fitting olmasını engeller.yani ezberin önüne geçer.
Max-Pooling işlemi yapılırken yukarıdaki gibi stride işleminde 1 er 1er kaydırma yapılmaz. kare kare kaydırılır.
bunun sonucunda ise Pooling Layer yapılmuş olur.
"""

"""FLATTENİNG İŞLEMİ 

Pool işleminden elde edilen sonucu,artificial nn mize sokabilmek için vektör haline getirme işlemidir. 
"""
"""FULLY CONNECTION İŞLEMİ
a-nn içeren kısımdır. Flattened edilmiş inputlar ANN e sokulur.Fully Connection işleminin özelliği tüm nodlar
diğer nodlarla bağlantılıdır.

"""
"""Convolutional kısmında filtreler öğrenilir."""
###CREATİNG MODEL WİTH KERAS
#conv-->maxpool-->dropout-->conv-->maxpool-->dropout-->full connected(2 layer)
#Dropout tekniği,bazı nöronların kapatılmasıdır.(fully connected işlemi esnasında).Böylece diğer nöronlarla bunlar arasında bağlantı kalmaz.
#Dropout overfittingi engellemek için önemli bir yoldur. 
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model=Sequential()


model.add(Conv2D(filters=8,kernel_size=(5,5),padding="Same",activation="relu",input_shape=(28,28,1)))
#8 tane filtre tanımladık. boyutları 5 e 5. padding işlemi için Same Padding methodunu kullanıyoruz.
model.add(MaxPool2D(pool_size=(2,2)))
#MaxPool işlemini yaptık. 
model.add(Dropout(0.25))
#4 de 1 nöron nodlarımızı kapatıyoruz.


model.add(Conv2D(filters=16, kernel_size=(3,3),padding="Same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

#fully connected

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10,activation="softmax")) #10 farklı outputumuz olabileceği için çıkış değerini 10 olarak belirledik.
#birden fazla class varsa yani multiple classification yapıyorsak softmax kullanılır. 

optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
epochs=25
batch_size=250

#Data Augmentation,bu sayede overfittingi önlemiş oluyoruz.Sonucu önemli derecede etkileyen parametrelerdir.
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

#Fitting the Model
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val), steps_per_epoch=x_train.shape[0] // batch_size)





