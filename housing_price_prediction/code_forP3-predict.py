# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:12:37 2019

@author: lanya
"""
import numpy as np
from copy import deepcopy

def grad_descent(xLst,yLst,a,max_itr):
    w=np.zeros(len(xLst[0]))
    for itr in range(1,max_itr+1):
        new_w=deepcopy(w)
        for j in range(len(w)):
            gradsum=0
            for i in range(len(yLst)):
                gradsum+=(np.dot(w,xLst[i])-yLst[i])*xLst[i][j]
            grad=gradsum/len(yLst)
            new_w[j]=w[j]-a*grad
        w=new_w
        loss=loss_function(w,xLst,yLst)
        print(itr,'Loss:',loss)
    return w

def loss_function(w,xLst,yLst):
    SE=0
    for i in range(len(yLst)):
        SE+=(yLst[i]-np.dot(w,xLst[i]))**2
    return SE/(2*len(yLst))

def predict(w,x,norm_const):
    for i in range(len(x)):
        mean,dev=norm_const[i]
        x[i]=(x[i]-mean)/dev
    x=[1]+x
    y=np.dot(w,x)
    mean,dev=norm_const[-1]
    return y

infile=open('normalized.txt','r')
NormConst=eval(infile.readline())
dataLst=infile.read().split('\n')[:-1]
x_array_lst,y_lst=[],[]
for data in dataLst:
    data=list(map(float,data.split(',')))
    x_array_lst.append(np.array([1]+data[:-1]))
    y_lst.append(data[-1])
infile.close()

#Gradient Descent
A,I=0.5,80
W=grad_descent(x_array_lst,y_lst,A,I)
print('\nLast and Minimum Loss:',loss_function(W,x_array_lst,y_lst))
print('Weight:',W)

#Predict by optimized w
X=[2650,4]
print('\nPrice of a house with %s square feet and %s bedrooms:'%tuple(X)\
      ,predict(W,X,NormConst))