# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:04:58 2020

@author: Vijay Sai Kondamadugu
"""

from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import numpy as np
import joblib

def pred_model(data, transforms):
    data.sex = transforms[0].transform(data.sex)
    data.smoker = transforms[1].transform(data.smoker)
    
    poly = transforms[3]
    model = transforms[4]
    X = data
    X_quad = poly.fit_transform(X)
    Ypred = model.predict(X_quad)
    
    return Ypred

app = Flask(__name__)

model = joblib.load("deployment_25062020.pkl") 
cols = ['age', 'sex', 'bmi', 'children', 'smoker']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = pred_model(data_unseen, model)
    prediction = int(prediction[0])
    return render_template('home.html', pred = 'Expected hospitalization charges {}'.format(prediction))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = pred_model(data_unseen, model)
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
    