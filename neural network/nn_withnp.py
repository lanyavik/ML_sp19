#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ML hw4 neural networks
from time import time
import numpy as np

def sigmoid(z_vector):
    a_vector=1/(1+np.exp(-z_vector)) #log_activate
    return a_vector

def softmax(z_vector):
    e_vector=np.exp(z_vector)
    a_vector=e_vector/np.sum(e_vector)
    return a_vector
    
def neural_network(x_h400,w_r25c401,w_r10c26):
    z_h25=np.dot(w_r25c401[:,1:],x_h400)+w_r25c401[:,0] #layer1 in
    a_h25=sigmoid(z_h25) #layer1 out
    z_h10=np.dot(w_r10c26[:,1:],a_h25)+w_r10c26[:,0] #layer2 in
    a_h10=softmax(z_h10) #layer2 out
    return a_h10

def classifier(image_vector,weights_1,weights_2):
    probs=neural_network(image_vector,weights_1,weights_2)
    predict=np.argmax(probs)+1
    return predict
#Q6 
#read data
def read_matrixdata(filename,size):
    infile = open(filename,'r')
    img_data = infile.read().strip().split('\n')
    img = [map(float,a.strip().split(',')) for a in img_data]
    pixels = []
    for p in img:
        pixels += p
    infile.close()
    return np.reshape(pixels,size)


# In[2]:


writings=read_matrixdata("ps5_data.csv",(5000,400))
labels=read_matrixdata("ps5_data-labels.csv",(5000,1))
L1weights=read_matrixdata("ps5_theta1.csv",(25,401))
L2weights=read_matrixdata("ps5_theta2.csv",(10,26))


# In[3]:


#classify
start_time=time()

predicts=[]
for digit in writings:
    num=classifier(digit,L1weights,L2weights)
    predicts.append(num)
    
print('Classify process runtime: %ss'%(time()-start_time))


# In[5]:


#check error rate
ercount=0
for idx in range(len(labels)):
    if labels[idx] != predicts[idx]:
        ercount+=1
print(ercount,'/ 5000')
print('Error Rate:',ercount/5000)


# In[ ]:




