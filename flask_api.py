from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger


app = Flask(__name__)
Swagger(app)

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome all'

@app.route('/predict', methods=['GET'])
def predict_note_authentication():
    
    '''Let's authenticate the Bank notes
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    '''
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return f'The predicted value is {prediction}'

@app.route('/predict_file', methods=['GET', 'POST'])
def predict_note_file_authentication():
    
    '''Let's authenticate the Bank notes
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
    '''
    
    df_test = pd.read_csv(request.files.get('file'))
    prediction = classifier.predict(df_test)
    return f'The predicted values for the csv is {list(prediction)}'


if __name__=='__main__':
    app.run(host="0.0.0.0", port=5000)
