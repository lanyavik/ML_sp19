# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 09:39:23 2019

@author: lanya
initiate weight to what?
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
    emailLst=datafile.readlines()
    datafile.close()
    Wordic=dict()
    for text in emailLst:
        for word in set(text[2:].split()):
            Wordic[word]=Wordic.get(word,0)+1
    #WORDS now is a dictionary of all words appeared and appearance time
    reptWordic=deepcopy(Wordic)
    for word,n in Wordic.items():
        if n < X:
            del reptWordic[word]
            
    reptWords=list(reptWordic.keys())
    print('Vocabulary length is',len(reptWords))        
    return emailLst,reptWords

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
            
def perceptron_train(emailLst,vocbLst,max_i):       
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
    print('...Training %s data with maximum iteration of %s times...'\
          %(len(emailLst),max_i))
    for i in range(1,max_i+1):
        e=0
        for y,x in yxLst:
            if y*array_dot(w,x) <= 0:
                w = array_sum(w,array_scl(y,x)) #updating weight
                k+=1 #record into total error
                e+=1 #record into iteration error
           #elif y*(np.dot(w,x)+w0) > 0: w=w (no change, no need to run)
        if e == 0: #sign of fit weight
           break
    print('Last error:',e,'Update:',k,'Passes:',i)
        #mechanism to halt the program if 100 passes did not find the fit wieght
        #so that I can check the variables, choose whether go on to plotting
    return (w,k,i)

split_data()
TRdata,Vocb=word("train.txt",26)
for I in [5,10,15]:
    Weight,Updates,Passes=perceptron_train(TRdata,Vocb,I)
    #check validation error
    validER=perceptron_error(Weight,"validation.txt",Vocb)
    print('Validation error percentage is',str(format(validER*100,'.1f'))+'%\n') 
print('Finish')
