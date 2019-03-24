# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 02:01:11 2019

@author:    neuroPanda
"""
from sklearn import linear_model 
lo_reg=linear_model.LogisticRegression(random_state=42,max_iter=150)
print("test accuracy: {}".format(lo_reg.fit(x_train.T,y_train.T).score(x_train.T,y_train.T)))
