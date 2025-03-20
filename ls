import numpy as np
import pandas as pd
import csv
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

# Create sample data for heart disease dataset
data = {
    'age': [63, 37, 41, 56, 57, 57, 56, 44, 52, 57],
    'sex': [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    'cp': [1, 2, 1, 1, 1, 1, 0, 2, 3, 1],
    'trestbps': [145, 130, 130, 120, 120, 140, 140, 120, 172, 150],
    'chol': [233, 250, 204, 236, 354, 192, 294, 263, 199, 168],
    'fbs': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'restecg': [2, 0, 2, 0, 0, 0, 2, 0, 1, 1],
    'thalach': [150, 187, 172, 178, 163, 148, 153, 173, 162, 174],
    'exang': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6, 0.4, 1.3, 0.0, 0.5, 1.6],
    'slope': [3, 3, 3, 2, 2, 2, 1, 2, 2, 2],
    'ca': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'thal': [6, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    'heartdisease': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
}

# Convert to DataFrame
heartDisease = pd.DataFrame(data)

# Save dataset to CSV file
heartDisease.to_csv('heart.csv', index=False)
print("CSV file 'heart.csv' created successfully!")

# Read the CSV file again
heartDisease = pd.read_csv('heart.csv')

# Display sample data
print('\nSample instances from the dataset:')
print(heartDisease.head())

# Display the attributes and datatypes
print('\nAttributes and datatypes:')
print(heartDisease.dtypes)

# Create Bayesian Network Model
model = BayesianModel([
    ('age', 'heartdisease'),
    ('sex', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])

# Learning CPDs using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum Likelihood Estimators')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Inferencing with Bayesian Network
print('\nInferencing with Bayesian Network:')
HeartDisease_test_infer = VariableElimination(model)

# Computing the Probability of Heart Disease given restecg
print('\n1. Probability of HeartDisease given evidence = restecg: 1')
q1 = HeartDisease_test_infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)

# Computing the Probability of Heart Disease given cp
print('\n2. Probability of HeartDisease given evidence = cp: 2')
q2 = HeartDisease_test_infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)