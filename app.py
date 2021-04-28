from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    pipe = pickle.load(f)


@app.route('/')
def index():
    return 'Index Page'

@app.route('/hello')
def hello_world():
    return 'Hello, World'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        the_data = request.get_json(force=True)
        new_data = the_data['new_data']
        # the following block of code is so the model can handle being fed one dimensional data
        new_data = np.array(new_data)
        if new_data.ndim == 1:
            new_data = [new_data]
        #back to the rest of the code    
        prediction_proba = pipe.predict_proba(new_data)
        prediction = prediction_proba[:, 1]
        return {'prediction': prediction.tolist()}
