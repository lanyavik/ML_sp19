# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 20:24:53 2019

@author: lanya
"""
import numpy as np

def normalize(xLst):
    mean=np.mean(xLst)
    standard_dev=np.std(xLst)
    norm_xLst=[]
    for x in xLst:
        norm_x=(x-mean)/standard_dev
        norm_xLst.append(norm_x)
        
    return norm_xLst,mean,standard_dev

infile=open('housing.txt','r')
dataLst=infile.read().split('\n')[:-1]
x1_lst,x2_lst,y_lst=[],[],[]
for data in dataLst:
    x1,x2,y=data.split(',')
    x1_lst.append(float(x1))
    x2_lst.append(float(x2))
    y_lst.append(float(y))
infile.close()

NormConst=[]
for var in ['x1_lst','x2_lst']:
    exec(var+',Mean,STdev=normalize(eval(var))')
    NormConst.append((Mean,STdev))
outfile=open('normalized.txt','w')
outfile.write(str(NormConst)+'\n')
for i in range(len(x1_lst)):
    outfile.write(str(x1_lst[i])+',')
    outfile.write(str(x2_lst[i])+',')
    outfile.write(str(y_lst[i])+'\n')
outfile.close()
        
    
