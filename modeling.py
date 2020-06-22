# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:27:07 2020

@author: Vijay Sai Kondamadugu
"""

from pycaret.regression import setup, create_model, plot_model, save_model, load_model
from pycaret.datasets import get_data

data = get_data('insurance')

r2 = setup(data, target = 'charges', session_id = 123,
           normalize = True,
           polynomial_features = True, trigonometry_features = True,
           feature_interaction=True, 
           bin_numeric_features= ['age', 'bmi'])

# Model Training and Validation 
lr = create_model('lr')

# plot residuals of trained model
plot_model(lr, plot = 'residuals')

# save transformation pipeline and model 
save_model(lr, model_name = 'C:/My work/Self_Projects/first_ml_deployment/deployment_21062020')


#deployment_21062020 = load_model('deployment_21062020')

