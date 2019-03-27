# -*- coding: utf-8 -*-
"""
Created on Sun Feb  12 17:41:38 2019

@author: lanya
--strange predictive parameters:
'deathtospamdeathtospamdeathtospam': 
"""
def find_mostpredictive(vocb,weight,nst):
    print('Most predictive parameters are')
    parameters=dict(zip(vocb,weight))
    for i in range(nst):
        maxi,mini=0,0
        for word,value in parameters.items():
            if value >= maxi:
                maxi=value
                maxword=word
            if value < mini:
                mini=value
                minword=word
        print("%sth positive word is '%s': %s"%(i+1,maxword,maxi))
        print("%sth negative word is '%s': %s"%(i+1,minword,mini))
        del parameters[maxword]
        del parameters[minword]

###Program starts here
vocbfile=open("vocabulary2677_rep26.txt",'r')
Vocb=eval(vocbfile.read())
vocbfile.close()
weightfile=open("weight2677_er0.021.txt",'r')
Weight=eval(weightfile.read())
weightfile.close()
find_mostpredictive(Vocb,Weight,12)
