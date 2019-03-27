# -*- coding: utf-8 -*-
"""
Created on Sun Feb  12 17:41:38 2019

@author: lanya

"""
from copy import deepcopy
#define some array operations
def array_dot(x,y):
    c=0
    for i in range(len(x)):
        c+=x[i]*y[i]
    return c
def array_sum(x,y):
    lst=[]
    for i in range(len(x)):
        lst.append(x[i]+y[i])
    return lst
def array_scl(c,x):
    lst=[]
    for i in range(len(x)):
        lst.append(c*x[i])
    return lst
import matplotlib.pyplot as plt

def word(dataname,X,N):
    datafile=open(dataname,"r")
    trainSET=datafile.readlines()[:N]
    Wordic=dict()
    print('...Buiding vocabulary by first %srows in "%s"...'%(N,dataname))
    for email in trainSET:
        for word in set(email[2:].split()):
            Wordic[word]=Wordic.get(word,0)+1
    datafile.close()
    #WORDS now is a dictionary of all words appeared and appearance time
    reptWordic=deepcopy(Wordic)
    for word,n in Wordic.items():
        if n < X:
            del reptWordic[word]
            
    reptWords=list(reptWordic.keys())
    print('Vocabulary length is',len(reptWords))
    return trainSET,reptWords

def feature_vector(email,vocbLst):
    x=[0]*len(vocbLst)
    for word in email.split():
        try:
            i=vocbLst.index(word)
            x[i]=1
        except ValueError:pass
    return x

def perceptron_error(w,validfile,vocbLst):
    datafile=open(validfile,"r")
    emailLst=datafile.readlines()
    datafile.close()
    wrong=0
    print('...Validating ...')
    for text in emailLst:
        if text[0] == '1':
            spam=True
        elif text[0] == '0':
            spam=False
        x=feature_vector(text[2:],vocbLst)
        spam_detect=(array_dot(w,x) >= 0)
        if spam != spam_detect:
            wrong+=1
    print ('Error rate: %s/%s'%(wrong,len(emailLst)))
    return wrong/len(emailLst)
            
def perceptron_train(emailLst,vocbLst):       
    w=[0]*len(vocbLst) #initiate w to be zero vector
    k,i=0,0
    yxLst=[]
    print('...Generating feacture vector...')
    for text in emailLst:
        info=[None,None]    
        if text[0]=='1':
            info[0]=1
        else:
            info[0]=-1
        info[1]=feature_vector(text[2:],vocbLst)
        yxLst.append(tuple(info))
    print('...Training %s data...'%len(emailLst))
    while True:
        e=0
        for y,x in yxLst:
            if y*array_dot(w,x) <= 0:
                w = array_sum(w,array_scl(y,x)) #updating weight
                k+=1 #record into total error
                e+=1 #record into iteration error
           #elif y*(np.dot(w,x)+w0) > 0: w=w (no change, no need to run)
        i+=1
        if e == 0: #sign of fit weight
           break
    print('\nFit:',(e==0),'Update:',k,'Passes:',i)
        #mechanism to halt the program if 100 passes did not find the fit wieght
        #so that I can check the variables, choose whether go on to plotting
    return (w,k,i)

###Program starts here
#loop to test throught all Ns
Q5data,Q6data=dict(),dict() #store requried data in dicts
for N in [200, 600, 1200, 2400, 3997]:
    TRset,vocb=word("train.txt",26,N)
    Weight,Updates,Passes=perceptron_train(TRset,vocb)
    ERrate=perceptron_error(Weight,"validation.txt",vocb)
    Q5data[N]=ERrate
    Q6data[N]=Passes
print('Finish',Q5data,Q6data,sep='\n')

#Plotting
'''Plot on seperate figures
'''
x1=list(Q5data.keys())
y1=list(Q5data.values())
#plottng as Question5 required
plt.plot(x1,y1,'bo',x1,y1)
plt.xlabel('amount of training data(N)')
plt.ylabel('validation error')   
plt.savefig("Q5plot.pdf")

plt.figure()
x2=list(Q6data.keys())
y2=list(Q6data.values())
#plotting as Question6 required
plt.plot(x2,y2,'bo',x2,y2)
plt.xlabel('amount of training data(N)')
plt.ylabel('number of perceptron iterations')
plt.savefig("Q6plot.pdf")
