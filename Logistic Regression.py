# -*- coding: utf-8 -*-
"""


@author: NeuroPanda
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Giriş için kullanacağımız verisetinde,x dosyasındakiler resimleri,y ler ise x teki dosyalara ait etiketleri ifade ediyor.
#Verisetindeki 204 ile 408 arası indexler 0ı,822 ile 1027 arası indexler 1i ifade eder.
x_l=np.load("X.npy")  #array tipinde verilerimizi load ettik. 
y_l=np.load("Y.npy")
img_size=64
plt.subplot(1,2,1)
plt.imshow(x_l[260].reshape(img_size,img_size))
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(x_l[900].reshape(img_size,img_size))


#Şimdi tüm resimlerimizi tek bir array içerisinde toplayalım.Büyük X olsun;
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0)#bu işlemle 410,64,64 şeklinde bir matris yaptık. Yani yukarıdan aşağıya 410 elemanı var.
#Aynı şekilde bunlara ait etiketleri de birleştirelim.
z=np.zeros(205)  #205 elemanı olan tek satır vektör oldu.
o=np.ones(205)   
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1) #Burada X arrayimizle uyumlu hale getirdik. 
#X arrayimiz 410 tane 64 e 64 lük pixellerden meydana geliyordu. 410 tanesi için sırayla 0 ve 1 atadık.



#Şimdi datasetimizi veriseti ve eğitim seti olarak ikiye böleceğiz.
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=42)
#%15 ini train set olarak ayırdık. random_stat ile de her seferinde aynı randomlıkta işlem yapacağız.
number_of_train=X_train.shape[0]  # X_train 348,64,64 lük bir matris haline geldi.
number_of_test=X_test.shape[0]


#Bundan sonraki aşamamız hem X hem Y vektörlerimizi aynı boyutlu hale getirmek .
X_train_flatten=X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[1])
X_test_flatten=X_test.reshape(number_of_test,X_test.shape[1]*X_test.shape[2])  #348,4096 lık bir hale getirdik.
#ŞİMDİ MATRİXLERİMİZİN TRANSPOZUNU ALALIM.
x_train=X_train_flatten.T  #4096,348 lik bir matris haline getirdik tekrar.Nottaki X haline geldi.
x_test=X_test_flatten.T
y_train=Y_train.T
y_test=Y_test.T
#İnitializing parameters
def initialize_weights_and_bias(dimension): #Bir öncekinden hatırlarsak 4096,348 lik içinde resimlerimiz bulunan bir 
    #array yapmıştık. yani tek bir resmimiz 496,1 lik bir matristi. Bunun için aynı şekilde 496,1 lik w matrisi yaptık.
    w=np.full((dimension,1),0.01) #496,1 lik bir matris oluşturacağız ve tüm değerleri 0.01 yapacağız.
    b=0.0
    return w, b 
w,b=initialize_weights_and_bias(x_train.shape[0])
#Sigmoid Function
def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head
#Forward and Backward Propagation 
  #Forward Propagation şöyle yapılır:
    #Weight ile X_traindeki resim matrislerimizi çarpacağız.
    #Çarpım sonucumuza göre bir z matrisi elde edeceğiz.
    #Elde ettiğimiz z matrisini loss func. sokacağız.
    #Loss funck. sonuçlarını toplayıp(z fonksiyonu içinde) cost fonk. değeri bulacağız.
def forward_backward_propagation(w,b,x_train,y_train):
    
    z=np.dot(w.T,x_train)+b  #Bu işlemleri bu şekilde kolaylaştırmaya vektörizasyon demiştik. 
    #Dikkat edelim z yi bir matris halinde,yani tek bir resimin çıktısı halinde değil tüm datasetin çıktısı halinde buluyoruz.
    y_head=sigmoid(z)
    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(y_head)
    cost=(np.sum(loss))/x_train.shape[1] #loss matrisindeki tüm loss değerlerini topladık ve ortalamasını aldık. 
    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] #type:ndarrayv size:4096 
    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]
    gradients={"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    return cost,gradients
#Bulduğumuz derivative_weight değerleri learning rate ile çarpılıp normal w değerlerinden düşülecek. Update işlemini ise aşağıda yapıyoruz.
#İmplementing Update Parameters
def update(w,b,x_train,y_train,learning_rate,number_of_iteration):
    cost_list=[]
    cost_list2=[]
    index=[]
    #number_of_iterations sayısı kadar parametrelerimizi güncelleyeceğiz.
    for i in range(number_of_iteration):
        cost,gradients=forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        w=w-learning_rate*gradients["derivative_weight"]
        b=b-learning_rate*gradients["derivative_bias"]
        if i%10==0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration {} : {}".format(i,cost))
    parameters={"weight": w, "bias": b}
    plt.plot(index,cost_list2)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters,gradients,cost_list


#Katsayılarımızı güncelledikten sonra prediction yapıyoruz. 
def predict(w,b,x_test):
    z=sigmoid(np.dot(w.T,x_test)+b) #tahmin yaparken forward propagationu uyguluyoruz.güncellenmiş parametrelerimizle bunu uyguladığımızda bize tahmin sonucunu verecek.
    y_prediction=np.zeros((1,x_test.shape[1]))
    for i in range(z.shape[1]):
        if z[0,i]<=0.5: #hesaplanan probabilite 0.5ten büyükse 1 dir değilse 0 dır diyoruz.
            y_prediction[0,i]=0
        else:
            y_prediction[0,i]=1
    
    return y_prediction


###########
##SON BİRLEŞTİRME!
###########
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):
    dimension=x_train.shape[0] #4096
    w,b=initialize_weights_and_bias(dimension)
    parameters,gradients,cost_list=update(w,b,x_train,y_train,learning_rate,num_iterations)
    y_prediction_test=predict(parameters["weight"],parameters["bias"],x_test)
    print("test accuracy: {}".format(100-np.mean(np.abs(y_prediction_test-y_test))*100))
 
    
logistic_regression(x_train,y_train,x_test,y_test,learning_rate=0.01,num_iterations=50)
 
 
 
 
 
 
 
 
