# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:03:22 2020

@author: I2ILAP-245
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


def train_model(df,y):
    x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.20)
    print(x_train.shape,y_train.shape)
    rf=RandomForestClassifier(n_estimators=100)
    rf.fit(x_train,y_train)
    filename = 'E:/finalized_model.pkl'
    pickle.dump(rf, open(filename, 'wb'))
    print("done")
    
    
def main():
    df=pd.read_csv("C:/Users/I2ILAP-245/Downloads/pima-indians-diabetes.csv",header=None)
    print(df.head())
    y=df[8]
    del df[8]
    train_model(df,y)
       

if __name__=="__main__":
    main()