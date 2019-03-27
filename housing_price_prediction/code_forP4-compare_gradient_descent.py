# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:12:37 2019

@author: lanya
"""
import numpy as np
from copy import deepcopy
from time import time

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
A,I=0.05,80
start_time=time()
W=grad_descent(x_array_lst,y_lst,A,I)
print('\nfunction "grad_descent" runtime: %ss'\
      %(time()-start_time))
print('Learning rate = %s, Passes = %s'%(A,I))
print('Last and Minimum Loss:',loss_function(W,x_array_lst,y_lst))
print('Weight:',W)
