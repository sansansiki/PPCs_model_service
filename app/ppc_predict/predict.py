'''
Author: gsl
Date: 2024-01-04 15:14:54
LastEditTime: 2024-01-12 18:10:16
FilePath: /workspace/model_service/app/ppc_predict/predict.py
Description: 

Copyright (c) 2024 by gsl, All Rights Reserved. 
'''
from ppc_predict import ppc_bp  
from flask import Flask, request, jsonify
import _pickle as pkl
import numpy as np
from . import data_load 
from flask_cors import CORS,cross_origin

# request
@ppc_bp.route('/predict', methods=['POST','GET'])
@cross_origin()

def predict():
    parameters = eval(str(request.json))
    
    X = data_load.load_data(parameters)
    # load model 
    y_pred_prob_list = []
    
    for i in range(1,6):
        with open(f'../model_service/ppcs_predict_model/XGBClassifier_{i}.pickle','rb') as f:
            clf_load = pkl.load(f)  
            y_pred_prob_list.append(clf_load.predict_proba(X)[:,1]) 
    # return json result
    return jsonify({'prediction': str(round(np.mean(y_pred_prob_list),3))})
