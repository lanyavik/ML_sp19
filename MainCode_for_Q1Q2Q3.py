# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 09:39:23 2019

@author: lanya
Main code for the perceptron
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

def split_data():
    datafile=open("spam_train.txt","r")
    trainfile=open("train.txt","w")
    validfile=open("validation.txt","w")
    n=1
    for line in datafile.readlines():
        if n <= 1000:
            validfile.write(line)
        else:
            trainfile.write(line)
        n+=1
    datafile.close()
    trainfile.close()
    validfile.close()

def word(dataname,X):
    datafile=open(dataname,"r")
    trainSET=datafile.readlines()
    Wordic=dict()
    print('...Buiding vocabulary by "%s"...'%dataname)
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
    outfile=open("vocabulary%s_rep%s.txt"%(len(reptWords),X),"w")
    outfile.write(str(reptWords))
    outfile.close()   
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
        i+=1
        for y,x in yxLst:
            if y*array_dot(w,x) <= 0:
                w = array_sum(w,array_scl(y,x)) #updating weight
                k+=1 #record into total error
                e+=1 #record into iteration error
           #elif y*(np.dot(w,x)+w0) > 0: w=w (no change, no need to run)
        print(i,':',e)
        if e == 0: #sign of fit weight
           break
    print('\nFit:',(e==0),'Update:',k,'Passes:',i)
        
    return (w,k,i)

split_data()
TRdata,Vocb=word("train.txt",26)
Weight,Updates,Passes=perceptron_train(TRdata,Vocb)
print('\nMistakes made before the algorithm terminates:',Updates)
#check if training error=0
if perceptron_error(Weight,"train.txt",Vocb) == 0:
    print('Training error is 0')
else:
    pause=input('Training error not 0!!!')
#validate
ERrate=perceptron_error(Weight,"validation.txt",Vocb)
print('Validation error percentage is',str(format(ERrate*100,'.1f'))+'%')
outfile=open("weight%s_er%s.txt"%(len(Weight),ERrate),"w")
outfile.write(str(Weight))
outfile.close() 
