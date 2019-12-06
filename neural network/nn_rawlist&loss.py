#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ML hw4 neural networks
from time import time
import math
#define dot product
def vec_dot(x,y):
    c=0
    for i in range(len(x)):
        c+=x[i]*y[i]
    return c

#Q1 function
def hidunit_process(x,w,b):
    z=vec_dot(w,x)+b #linear_model
    a=1/(1+math.exp(-z)) #log_activate
    return a
#Q2 function
def softmax(z_vector):
    e_vector=[]
    for z in z_vector:
        e_vector.append(math.exp(z))
    a_vector=[]
    for e in e_vector:
        a_vector.append(e/sum(e_vector))
    return a_vector
#Q3 function       
def outlayer_process(x,wLst,bLst):
    a_vector=[]
    for i in range(len(bLst)):
        a_vector.append(vec_dot(x,wLst[i])+bLst[i])
    return softmax(a_vector)
#Q4 function:
def hidlayer_process(x,wLst,bLst):
    a_vector=[]
    for i in range(len(bLst)):
        a_vector.append(hidunit_process(x,wLst[i],bLst[i]))
    return a_vector
def neural_network(x_h400,w_r25c401,w_r10c26):
    a_h25=hidlayer_process(x_h400,list(map(list, zip(*w_r25c401[1:]))),w_r25c401[0])
    a_h10=outlayer_process(a_h25,list(map(list, zip(*w_r10c26[1:]))),w_r10c26[0])
    return a_h10
#Q5 function
def classifier(image_vector,weights_1,weights_2):
    probs=neural_network(image_vector,weights_1,weights_2)
    predict=probs.index(max(probs))+1
    return predict
#Q6 
#read data
def read_matrixdata(filename,order):
    infile = open(filename,'r')
    img_data = infile.read().strip().split('\n')
    row_matrix = [list(map(float,a.strip().split(','))) for a in img_data]
    col_matrix=list(map(list, zip(*row_matrix)))
    return eval('%s_matrix'%order)


# In[2]:


writings=read_matrixdata("ps5_data.csv",'row')
labels=read_matrixdata("ps5_data-labels.csv",'col')[0]
L1weights=read_matrixdata("ps5_theta1.csv",'col')
L2weights=read_matrixdata("ps5_theta2.csv",'col')


# In[3]:


#classify
start_time=time()

predicts=[]
for digit in writings:
    num=classifier(digit,L1weights,L2weights)
    predicts.append(num)
    
print('Classify process runtime: %ss'%(time()-start_time))


# In[7]:


#check error rate
ercount=0
for idx in range(len(labels)):
    if labels[idx] != predicts[idx]:
        ercount+=1
print(ercount,'/ 5000')
print('Error Rate:',ercount/5000)


# In[10]:


# Q7 evaluate cross entropy loss
def cross_entropy(images,weights_1,weights_2,ylabels):
    loss=0
    for image_vector,y in zip(images,ylabels):
        probs=neural_network(image_vector,weights_1,weights_2)
        loss+=math.log(probs[int(y-1)])
    loss= -loss/len(ylabels)
    return loss
       
print('Cross Entropy Loss of the given weights:', cross_entropy(writings,L1weights,L2weights,labels))


# In[ ]:




