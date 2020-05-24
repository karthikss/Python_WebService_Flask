# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:37:43 2020

@author: I2ILAP-245
"""

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from flask import Flask,request,jsonify
from flask_cors import CORS,cross_origin

rf_model=pickle.load(open("E:/diab_pickle.pkl",'rb'))
print(rf_model)
headers=['A','B','C','D','E','F','G','H']
input_variables=pd.DataFrame([[1,4,2,3,5,6,7,8]],columns=headers,index=['input'],dtype=float)
print(input_variables)
resf=rf_model.predict(input_variables)
resp=rf_model.predict_proba(input_variables)
print(resf)
print(resp)
app=Flask(__name__)
CORS(app)
@app.route("/diabetes/predict",methods=['POST'])

def predict:
    payload_list=request.json['mydata']
    result={}
    for payload in payload_list:
        for key,values in payload.item():
            value=[float(i) for i in values.split(',')]
            input_variables=pd.DataFrame([value],column=headers,index=['input'],dtype=float)
            res=rf_model.predict_proba(input_variables)
            result[key]=[]
            result[key].append(str(res))
    print(result)
    return result

if __name__=='__main__':
    app.run(debug=False,host='127.0.0.1',port=8000)
        
