# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:33:41 2024

@author: Admin
"""

from flask import Flask,request
import joblib
import pandas as pd

model=joblib.load(r'./bank_model.pkl')
gen_encoder=joblib.load('./gen_encoder.pkl')
geo_encoder=joblib.load('./geo_encoder.pkl')


app=Flask(__name__)

@app.route("/",methods=['GET','POST'])

def predict():
    data=request.get_json(force=True)
    print(data)
    data=data["info"]
    data=pd.DataFrame([data])
    data['Gender']=gen_encoder.transform(data['Gender'])
    geo_data=geo_encoder.transform(data[['Geography']])
    geo_data=pd.DataFrame(geo_data.toarray(),columns=geo_encoder.get_feature_names_out())
    newdata=pd.concat([geo_data,data],axis='columns')
    newdata=newdata.drop('Geography',axis='columns')
    out=model.predict(newdata)

    
    return str(out)

app.run(host='0.0.0.0')