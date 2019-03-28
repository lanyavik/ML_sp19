#!/usr/bin/env python
# coding: utf-8

# In this exercise, you will be asked to implement the K-Nearest Neighbor (KNN) algorithm on the MNIST dataset.

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import time
from numpy import linalg as LA


# In[2]:


### Load the dataset
train_set = np.loadtxt('mnist_train.csv', delimiter=',')
test_set = np.loadtxt('mnist_test.csv', delimiter=',')
x_train, y_train = train_set[:,1:], train_set[:,0].astype(int)
x_test, y_test = test_set[:,1:], test_set[:,0].astype(int)
num_train, num_test = train_set.shape[0], test_set.shape[0]


# In[3]:


# Display image
def show_image(feature_matrix, index):
    '''
    Displays one of the digits in the dataset
    Input: Feature matrix
    Output: grayscale image of digit
    '''
    image_vector = feature_matrix[index]
    image_matrix = np.array(image_vector).reshape(28,28)
    plt.gray()
    plt.imshow(image_matrix, interpolation='nearest')
    plt.show()


# In[6]:


class KNN:
    
    def __init__(self, x_train, y_train, K):
        self.x_train = x_train
        self.y_train = y_train
        self.K = K
        
    def compute_distance_two_loop(self, x_test, y_test):
        num_train = self.x_train.shape[0]
        num_test = x_test.shape[0]
        dist = np.zeros((num_test, num_train))
        
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dist[i, j].               #
                #####################################################################
                
                dist[i,j]=LA.norm(x_test[i]-self.x_train[j])
                pass
                ########################### End of Your Code ########################
                #####################################################################
            
        return dist
    
    
    
    
    def compute_distance_one_loop(self, x_test, y_test):
        num_train = self.x_train.shape[0]
        num_test = x_test.shape[0]
        dist = np.zeros((num_test, num_train))
        
        for i in range(num_test):
            #####################################################################
            # TODO:                                                             #
            # Compute the l2 distance between the ith test point and the all    #
            # training point, and store the result in dist[i, :].              #
            #####################################################################
            
            dist[i,:]=LA.norm(self.x_train-x_test[i],axis=1)
            pass
            ########################### End of Your Code ########################
            #####################################################################
            
        return dist
    
    def predict(self, dist):
        num_test = dist.shape[0]
        y_pred = np.zeros(num_test)
        
        #####################################################################
        # TODO:                                                             #
        # Using the dist matrix, calculate the prediction results for the   #
        # test set and store in y_pred (hint: look up np.argsort)           #
        #####################################################################
            
        for i in range(num_test):
            idx=np.argsort(dist[i,:])
            k_nearest=self.y_train[idx][:self.K]
            y_pred[i]=np.bincount(k_nearest).argmax()
        pass
        ########################### End of Your Code ########################
        #####################################################################
            
        return y_pred
    
    
    def get_accuracy(self, y_pred, y_test):
        #################################################
        # TODO: Calculate accuracy of KNN classifier on #
        # on test set.                                  #
        #################################################
        
        acucount=0
        for pred,test in zip(y_pred, y_test):
            if pred == test:
                acucount+=1
        accuracy=acucount/len(y_pred)
        pass
        ################################################
        ############# End of Your Code #################
        ################################################
        return accuracy


# In[7]:


# Set hyperparameter
K = 5

knn = KNN(x_train, y_train, K)

# Using two loops
start = time.time()
dist = knn.compute_distance_two_loop(x_test, y_test)
end = time.time()
print("Computing the distance matrix using two loops took ", end-start)

# Using one loop
start = time.time()
dist = knn.compute_distance_one_loop(x_test, y_test)
end = time.time()
print("Computing the distance matrix using one loop took ", end-start)

# Predict results
y_pred = knn.predict(dist)

# Get accuracy
accuracy = knn.get_accuracy(y_pred, y_test)


print('My classifier has an accuracy of', accuracy)



# In[ ]:




