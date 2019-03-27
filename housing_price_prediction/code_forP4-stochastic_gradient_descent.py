# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:12:37 2019

@author: lanya
"""
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time

def stocha_grad_descent(xyPairs,a,max_itr):
    w=np.zeros(len(xyPairs[0][0]))
    loss_record=dict()
    for itr in range(1,max_itr+1):
        xLst,yLst=zip(*xyPairs)
        new_w=deepcopy(w)
        for i in range(len(yLst)):
            for j in range(len(w)):
                grad=(np.dot(w,xLst[i])-yLst[i])*xLst[i][j]
                new_w[j]=w[j]-a*grad
            w=new_w #update weight
            loss=loss_function(w,xLst,yLst)
            print(itr,i,'Loss:',loss)
            loss_record[(itr-1)*len(yLst)+i+1]=loss
        random.shuffle(xyPairs)
        
    return (w,loss_record)

def loss_function(w,xLst,yLst):
    SE=0
    for i in range(len(yLst)):
        SE+=(yLst[i]-np.dot(w,xLst[i]))**2
    return SE/(2*len(yLst))

infile=open('normalized.txt','r')
NormConst=eval(infile.readline())
dataLst=infile.read().split('\n')[:-1]
x_array_lst,y_lst=[],[]
for data in dataLst:
    data=list(map(float,data.split(',')))
    x_array_lst.append(np.array([1]+data[:-1]))
    y_lst.append(data[-1])
infile.close()
Xy=list(zip(x_array_lst,y_lst))

#Stochastic Gradient Descent
A,I=0.05,3
start_time=time()
W,Record=stocha_grad_descent(Xy,A,I)
print('function "stocha_grad_descent" runtime: %ss'\
      %(time()-start_time))
print('Last Loss:',loss_function(W,x_array_lst,y_lst))
print('Weight:',W)
#Plot J(w) on each update
i=np.array(list(Record.keys()))
Jw=np.array(list(Record.values()))
plt.plot(i,Jw)
plt.title('Stochastic Gradient Descent of %s passes'%I)
plt.text(30,4e10,'learning rate = '+str(A))
plt.xlabel('number of updates')
plt.ylabel('Loss function J(w)')
plt.show()