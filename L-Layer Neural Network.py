# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 01:09:12 2019

@author: NeuroPanda
"""
import numpy as np

#from numpy import numpy.core.__multiarray__umath
x_l=np.load("X.npy")
y_l=np.load("Y.npy")
x=np.concatenate((x_l[204:409], x_l[822:1027] ),axis=0)
z=np.zeros((1,205))
o=np.ones((1,205))
y=np.concatenate((z,o),axis=0).reshape(x.shape[0],1)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.15,random_state=42)
x_train_flatten=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[1])
x_test_flatten=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[1])
#x_train_flatten2=x_train.reshape(x_train.shape[1]*x_train.shape[1],x_train.shape[0]) #Dikkat!Bu ikisi aynı şey değildir.
x_train=x_train_flatten.T
x_test=x_test_flatten.T
y_train=Y_train.T
y_test=Y_test.T

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library

classifier = Sequential() # initialize neural network
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim=4096))
#classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
classifier.fit(x_train.T,y_train.T,epochs=100,batch_size=32)


score=classifier.evaluate(x_test.T,y_test.T,batch_size=32)

print(score)




