# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 03:29:49 2019

@author: NeuroPanda
"""
import numpy as np
kalp_data=heartcsv.T
y_all=kalp_data[13,:].reshape(-1,1)
x_all=kalp_data[0:13,:]
y_train1=y_all[0:120,:]
y_train2=y_all[165:265,:]
y_train=np.concatenate((y_train1,y_train2),axis=0)
x_train1=x_all[:,0:120]
x_train2=x_all[:,165:265]
x_train=np.concatenate((x_train1,x_train2),axis=1)
x_test1=x_all[:,120:165]
x_test2=x_all[:,265:303]
x_test=np.concatenate((x_test1,x_test2),axis=1)
y_test1=y_all[120:165,:]
y_test2=y_all[265:303,:]
y_test=np.concatenate((y_test1,y_test2),axis=0)

y_train=y_train.T
y_test=y_test.T
def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head
def initializing_parameters_and_layer_sizes_NN(x_train,y_train):
    parameters={"weight1":np.random.randn(3,x_train.shape[0])*0.01,
                "bias1":np.zeros((3,1)),
                "weight2":np.random.randn(y_train.shape[0],3)*0.01,
                "bias2":np.zeros((y_train.shape[0],1))}
    return parameters
def forward_propagation(x_train,parameters):
    Z1=np.dot(parameters["weight1"],x_train)+parameters["bias1"]
    A1=np.tanh(Z1)
    Z2=np.dot(parameters["weight2"],A1)+parameters["bias2"]
    A2=sigmoid(Z2)
    cache={"Z1":Z1,
              "A1":A1,
              "A2":A2,
              "Z2":Z2}
    return A2,cache
def compute_cost_NN(A2,Y):
    logprobs=np.multiply(np.log(A2),Y)  
    cost=-np.sum(logprobs)/Y.shape[1]
    return cost
def backward_propagation_NN(parameters, cache, X, Y):    
#ilk dersten hatırlarsak, update yapabilmek için bize weight değelerine göre türev değerleri lazımdı. 
#bunları bulmak için de önce dz2 ve dz1 değerlerini buluyoruz.Daha sonra update edebilmek için bulduğumuz w be b değerlerini kullanacağız.
    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1] 
    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    return grads
def update_parameters(parameters,grads,learning_rate=0.2):
    parameters={"weight1":parameters["weight1"]-learning_rate*grads["dweight1"],
                "weight2":parameters["weight2"]-learning_rate*grads["dweight2"],
                "bias1":parameters["bias1"]-learning_rate*grads["dbias1"],
                "bias2":parameters["bias2"]-learning_rate*grads["dbias2"]}
    return parameters

def prediction(x_test,parameters):
    A2,cache=forward_propagation(x_test,parameters)
    Y_prediction=np.zeros((1,x_test.shape[1]))
    for i in range(A2.shape[1]):
        if A2[0,i]<=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
    return Y_prediction

def two_layer_neural_network(x_train,y_train,x_test,y_test,num_iterations):
    parameters=initializing_parameters_and_layer_sizes_NN(x_train,y_train)
    cost_list=[]
    for i in range(0,num_iterations):
        A2,cache=forward_propagation(x_train,parameters)
        cost=compute_cost_NN(A2,y_train)
        cost_list.append(cost)
        grads=backward_propagation_NN(parameters,cache,x_train,y_train)
        parameters=update_parameters(parameters,grads,learning_rate=0.2)
        
    y_prediction=prediction(x_test,parameters)
    accuracy=(100-np.mean(np.abs(y_prediction-y_test))*100)
    return accuracy

liste=[]
sayi=1
two_layer_neural_network(x_train,y_train,x_test,y_test,2000)

for i in range(1,30000):
    accuracy=two_layer_neural_network(x_train,y_train,x_test,y_test,i)
    liste.append(accuracy)
    print("process {}".format(sayi))
    sayi+=1
    print(max(liste))
    


    