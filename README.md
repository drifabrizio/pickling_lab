# **Pickling Lab**

## Description

This lab shows how to pickle a fitted model so that it may be used later. This repository contains:
* A requirements.txt file containing all the necessary packages for this lab
* A Jupyter notebook containing the code for:
    * Importing the data
    * Fitting a pipeline
    * Pickling the fitted pipeline
    * Creating new data to test the model on      
* A file containing the pickled model
* A Python file containing the new data
* A Python file for testing the pickled model with the new data

## Launching a Jupyter notebook

In a shell with your virtual environment running, use pip install to install packages needed.

Using the requirements.txt file:

```bash
pip install requirements.txt
```

Installing the packages individually:

```bash
pip install pywin32
pip install sklearn
pip install notebook
pip install pandas
```
Finally launch the Jupyter notebook with:

```bash
Jupyter notebook
```
## Building and pickling a model

Note: For this lab, I chose a [diabetes dataset](https://www.openml.org/data/get_csv/22044302/) from openml.

Import your dataset:

```bash
import pandas as pd
df = pd.read_csv('https://www.openml.org/data/get_csv/22044302/')
```

### Fitting a pipeline

Import neccessary packages and functions from sklearn:

```bash
from sklearn.ensemble import RandomForestClassifie
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
```

Make training and testing data:

```bash
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(df.drop('Outcome', axis=1), df.Outcome, random_state=0)
```
**Note: based on the dataset I chose, I decided to fit a RandomForestClassifier model, but these steps apply to other types of models.**

Make the pipeline:

```bash
pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
pipe.fit(X_train, y_train)
Pipeline(steps=[('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
```

### Pickle the Model

```bash
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(pipe, f, pickle.HIGHEST_PROTOCOL)
```

## Making new data

Here we will use json to create new data and save it to a new file

``` bash
import json

new = df.drop('Outcome', axis=1) # our new data needs be only our independent variables

with open('newdata.py', 'w') as f:
    json.dump(new.iloc[0].values.tolist(), f)
```

## Testing the model

Make a new Python file to test the model.

**Note: I made mine in my project directory in Atom and I named it test_pickled_model.py**

In the new Python file write:

```bash
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
```

Finally, in a shell with your virtual environment running, run the following command to test the model:

```bash
python test_pickled_model.py
```
