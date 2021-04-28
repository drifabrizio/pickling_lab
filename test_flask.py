import requests
import json
import numpy as np

#load test data
with open('newdata.py', 'r') as f:
    newdata = json.load(f)

#newdata = np.array(newdata)
#if newdata.ndim == 1:
#   newdata = [newdata]


r = requests.post('http://127.0.0.1:5000/predict', json={'new_data': newdata})
data = r.json()
prediction = data['prediction']
print('Good day to you. The prediction for the data is:', prediction)
