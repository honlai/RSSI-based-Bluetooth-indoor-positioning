# ⚙️ Model Usage & Evaluation Guide

This document provides a clear example of how to load the dataset, train multiple models, make predictions, and evaluate performance on the test set.

## 📦 1. Load Data

```python
from utils import load_data

# loading data
df_X_train, df_y_train, df_X_test, df_y_test = load_data()

X_train = df_X_train
y_train_coor = df_y_train[['x', 'y']]
y_train_id = df_y_train['cell_id']

X_test = df_X_test
y_test_coor = df_y_test[['x', 'y']]
y_test_id = df_y_test['cell_id']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## 🧩 2. Train & Predict

Each model is implemented as a self-contained class or function with a simple `fit()` and `predict()` interface.

```python
# Example: Import models
from models.LightGBM import LightGBM
from models.linear_discriminant_analysis import linear_discriminant_analysis
from models.Logistic import Logistic

# Initialize models
models = [LightGBM(),linear_discriminant_analysis(),Logistic()]

for model in models:
    print(f'Fitting {model.name}-------------------------------')
    model.fit(X_train=X_train, df_y_train=df_y_train)
    y_pred_id, y_pred_coor = model.predict(X_test)
```

## 🎯 3. Evaluate Models

Below is an example of how to compute basic evaluation metrics (e.g., mean positioning error) and summarize the results in a table.

```python
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from utils.evaluation import compute_distances_error

results = {'distance_error': dict(), "accuracy": dict(
), 'max_distance': dict(), 'min_distance': dict()}
accuracy = accuracy_score(y_test_id, y_pred_id)
dis_error, max_dis, min_dis = compute_distances_error(y_test_coor, y_pred_coor)
results['distance_error'][model.name] = dis_error
results['accuracy'][model.name] = accuracy
results['max_distance'][model.name] = max_dis
results['min_distance'][model.name] = min_dis
```

## 📊 4. Result Table

|Model| Mean Positioning Error (m) |cell prediction accuracy |max distance error|min distance error|
|-----| -------------------------- |-------------------------|------------------|------------------|
| AdaBoost      | 9.416312                   |0.004  | 21.936217  |  0.566265|
| LogisticRegression | 2.631670              | 0.188 | 19.637457| 0.116166|
