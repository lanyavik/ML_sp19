# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:12:37 2019

@author: lanya
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def grad_descent(xLst,yLst,a,max_itr):
    w=np.zeros(len(xLst[0]))
    loss_record=dict()
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
        if itr%10 == 0:
            loss_record[itr]=loss
    return (w,loss_record)

def loss_function(w,xLst,yLst):
    SE=0
    for i in range(len(yLst)):
        SE+=(yLst[i]-np.dot(w,xLst[i]))**2
    return SE/(2*len(yLst))

#The program starts here
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
for A in [0.01,0.03,0.1,0.2,0.5]:
    print('\nLearning rate=',A)
    W,Record=grad_descent(x_array_lst,y_lst,A,80)
    print('Last and Minimum Loss:',loss_function(W,x_array_lst,y_lst))
    #plot
    i=np.array(list(Record.keys()))
    Jw=np.array(list(Record.values()))
    plt.plot(i,Jw)
plt.xlabel('number of iterations')
plt.ylabel('loss J(w)')
plt.legend(('learning rate a=0.01','learning rate a=0.03',\
            'learning rate a=0.1','learning rate a=0.2',\
            'learning rate a=0.5'),loc='upper right')
plt.savefig('P2.B(b,c)-plot_5LR.pdf')
'''
a=0.03 gives the best result, because J(w) descents relatively fast
and converges at a value close to its minima (relatively optimized)
'''
