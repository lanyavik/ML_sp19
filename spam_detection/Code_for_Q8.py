# -*- coding: utf-8 -*-
"""
Created on Wen Feb  13 20:34:38 2019

@author: lanya
--changed configuration X in[22,26,28];BIAS in [0,0.1,0.2]
haven't considered I  what is the meaning?
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
    return trainSET,reptWords

def feature_vector(email,vocbLst):
    x=[0]*len(vocbLst)
    for word in email.split():
        try:
            i=vocbLst.index(word)
            x[i]=1
        except ValueError:pass
    return x

def perceptron_error(w,filename,vocbLst):
    datafile=open(filename,"r")
    emailLst=datafile.readlines()
    datafile.close()
    wrong=0
    print('...Validating by "%s" ...'%filename)
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
    print('...Training %s data with iteration upperlimit of %s times...'\
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
    print('Fit:',(e==0),'Update:',k,'Passes:',i)
        #mechanism to halt the program if 100 passes did not find the fit wieght
        #so that I can check the variables, choose whether go on to plotting
    return (w,k,i)

###Program starts here
'''#loop to test throught all Ns'''

Q8data=dict()
for X in [22,26,28]:
    TRset,Vocb=word("train.txt",X)
    for I in [5,10,15]:
        config=(X,I)
        print('\nRep %s words; max %s iterations'%config)
        Weight,Updates,Passes=perceptron_train(TRset,Vocb,I)
        validER=perceptron_error(Weight,"validation.txt",Vocb)
        testER=perceptron_error(Weight,"spam_test.txt",Vocb)
        Q8data[config]=(validER,testER)
print('Report:\nX\t\tmaxIter\t  validation error  (test error)')
for key,value in Q8data.items():
    print(key[0],key[1],value[0],value[1],sep='\t\t')
#Output:

print('Manually select the best configuration:')
while True:
    X,I=eval(input('Enter configuration data(X)>')) 
    print('try Vocabulary rep at least %s, Iteration upperlimit %s'%(X,I))
    #Test weight on best configuration
    TRset,Vocb=word("train.txt",X)
    Weight,Updates,Passes=perceptron_train(TRset,Vocb,I)
    ERrate=perceptron_error(Weight,"spam_test.txt",Vocb)
    print('Test error percentage is',str(format(ERrate*100,'.1f'))+'%')
#Output:
