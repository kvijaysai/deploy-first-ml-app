# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:23:04 2020

@author: Vijay Sai Kondamadugu
"""

# import numpy as np 
import pandas as pd 
# import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import joblib


data = pd.read_csv('insurance.csv')


#sex
sex_le = LabelEncoder()
sex_le.fit(data.sex.drop_duplicates()) 
data.sex = sex_le.transform(data.sex)
# smoker or not
sm_le = LabelEncoder()
sm_le.fit(data.smoker.drop_duplicates()) 
data.smoker = sm_le.transform(data.smoker)
#region
re_le = LabelEncoder()
re_le.fit(data.region.drop_duplicates()) 
data.region = re_le.transform(data.region)

X = data.drop(['charges','region'], axis = 1)
Y = data.charges

quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X)

# X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)

# plr = LinearRegression().fit(X_train,Y_train)
plr = LinearRegression().fit(x_quad,Y)

req_pkl = [sex_le, sm_le, re_le, quad, plr]
# save
joblib.dump(req_pkl, "deployment_25062020.pkl") 

# Y_train_pred = plr.predict(X_train)
# Y_test_pred = plr.predict(X_test)

# print(plr.score(X_test,Y_test))

#testing
data2 = pd.read_csv('insurance.csv')
model = joblib.load("deployment_25062020.pkl") 

def pred_model(data, transforms):
    data.sex = transforms[0].transform(data.sex)
    data.smoker = transforms[1].transform(data.smoker)
    data.region = transforms[2].transform(data.region)
    
    poly = transforms[3]
    model = transforms[4]
    X = data.drop(['region'], axis = 1)
    X_quad = poly.fit_transform(X)
    Ypred = model.predict(X_quad)
    
    return Ypred

cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
inp = pd.DataFrame([data2.loc[0]], columns = cols)
result = pred_model(inp, model)
print(int(result[0]))
