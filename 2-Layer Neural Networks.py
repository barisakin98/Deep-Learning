# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:53:24 2019

@author: NeuroPanda
"""
import numpy as np

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
def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head

#İNİTALİZİNG PARAMETERS AND LAYER SİZES
#tanh fonksiyonunun yapısından dolayı, weightler çok büyük değerler alırlarsa,türevleri sıfıra yaklaşacağı için
#backward propagation yapmak zorlaşacaktır. Bu nedenle ortalama değerler vermek en doğrusudur.

def initialize_parameters_and_layer_sizes_NN(x_train,y_train):
    parameters={"weight1":np.random.randn(3,x_train.shape[0])*0.1,  #Normalde 1,4096lık bir weight yaparken,3nodumuz olacağı için burada 3,4096 diyoruz.
                "bias1":np.zeros((3,1)),#aynı şekilde 3 nodumuz olduğu için 3 tane satırımız var. 
                "weight2":np.random.randn(y_train.shape[0],3)*0.1,   #A nın size ı 3,1 olacağı için çarpabilmek adına 1 e 3lük matris belirledik.
                "bias2":np.zeros((y_train.shape[0],1))} #aynı şekilde toplama için 1,1 lik matris yaptık. 
    return parameters

##FORWARD PROPAGATİON 
def forward_propagation_NN(x_train,parameters):
    Z1=np.dot(parameters["weight1"],x_train)+parameters["bias1"]
    A1=np.tanh(Z1)#hidden layerimizin içeriğini elde etmiş olduk. yani hidden layerdan bir sonraki layer a geçecek inputu hesaplamış olduk.
    Z2=np.dot(parameters["weight2"],A1)+parameters["bias2"]
    A2=sigmoid(Z2) #1 ile 0 arasında probabilistik bir değer elde ettik. yani y_head değerimiz bu.
    cache={"A1":A1,
           "Z1":Z1,
           "Z2":Z2,
           "A2":A2}
    return A2,cache #A2 dediğimiz şey aslında bizim y_head değerimiz!!!!#Daha sonra backward propagation yaparken bu A1,Z1 değerleri işimize yarayacağı için cache de depoladık.

#LOSS AND COST FUNCTİONS
    #loss fonksiyonu olarak ANN lerde Cross Entropi fonksiyonu kullanılır. Logistic regressionda kullandığımızdan farklıdır.
def compute_cost_NN(A2,Y,parameters):
    logprobs=np.multiply(np.log(A2),Y)  #Y yerine ileride y_train parametresi gelecek.
    cost=-np.sum(logprobs)/y.shape[1]
    return cost
#BACKWARD PROPAGATİON 
    #TÜREV ALARAK GERİYE DOĞRU GİDİYORUZ. COST FONKSİYONUNA GÖRE ÖNCE Z2NİN TÜREVİNİ,DAHA SONRA DA Z1İN TÜREVİNİ BULACAĞIZ.
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
#UPDATİNG PARAMETERS
def update_parameters_NN(parameters,grads,learning_rate=0.01):
    parameters={"weight1":parameters["weight1"]-learning_rate*grads["dweight1"],
                "weight2":parameters["weight2"]-learning_rate*grads["dweight2"],
                "bias1":parameters["bias1"]-learning_rate*grads["dbias1"],
                "bias2":parameters["bias2"]-learning_rate*grads["dbias2"]}
    return parameters
#PREDİCTİON
def prediction(x_test,parameters):
    A2,cache=forward_propagation_NN(x_test,parameters)
    Y_prediction=np.zeros((1,x_test.shape[1]))
    for i in range(A2.shape[1]):
        if A2[0,i]<=0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
    return Y_prediction
#CREATE MODEL
def two_layer_neural_network(x_train,y_train,x_test,y_test,num_iterations):
    parameters=initialize_parameters_and_layer_sizes_NN(x_train,y_train)
    cost_list=[]
    for i in range(0,num_iterations):
            A2,cache=forward_propagation_NN(x_train,parameters)
            cost=compute_cost_NN(A2,y_train,parameters)
            cost_list.append(cost)
            grads=backward_propagation_NN(parameters,cache,x_train,y_train)
            parameters=update_parameters_NN(parameters,grads,learning_rate=0.01)
            
    y_prediction=prediction(x_test,parameters)
    accuracy=(100-np.mean(np.abs(y_prediction-y_test))*100)
   #print("Accuracy=:{} ".format(accuracy))
    #print(min(cost_list),max(cost_list))
    #print(accuracy)
    return accuracy

print(two_layer_neural_network(x_train,y_train,x_test,y_test,2500))


"""liste=[]
sayi=1
for i in range(2000,4000):
    accuracy=two_layer_neural_network(x_train,y_train,x_test,y_test,i)
    liste.append(accuracy)
    print("process {}".format(sayi))
    sayi+=1
    print(max(liste))"""





    

    
            
      
  

        
    
    




































