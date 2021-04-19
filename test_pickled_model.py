import json
import pickle
import numpy as np

with open('newdata.py', 'r') as f:
    newdata = json.load(f)

newdata = np.array(newdata)
if newdata.ndim == 1:
    newdata = [newdata]

with open('model.pkl', 'rb') as f:
    pipe = pickle.load(f)

predictions = pipe.predict_proba(newdata)
print(predictions)
