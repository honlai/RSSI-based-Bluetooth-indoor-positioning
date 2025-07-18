# 📡 RSSI-based Indoor Positioning System

This project provides a **simulated dataset**, **baseline models**, and **evaluation scripts** for predicting the 2D location of an object based on **RSSI signals** from multiple sensors.  
It is designed as a research prototype for indoor positioning in an obstacle-free factory scenario.

---

## 🚩 Project Overview

- **Objective:** Estimate the precise (x, y) location of a moving object based on RSSI values received by multiple fixed sensors (anchors).
- **Environment:** A 20m × 24m unobstructed factory floor divided into 1m × 1m grid cells.
- **Dataset:** Includes training samples generated from grid cell centers with Gaussian noise and test samples generated randomly.
- **Sensors:** 11 fixed anchors whose coordinates are provided.

---

## 📂 Repository Structure

```bash
RSSI-based-Bluetooth-indoor-positioning/
│
├── dataset/
│ ├── anchors_pos.csv
│ ├── centers_rssi.csv
│ ├── centers_pos_cell.csv
│ ├── tests_rssi.csv
│ └── tests_pos_cell.csv
│
├── images/
│ ├── center_points_plot.png
│ └── test_points_plot.png
│
├── models/
│ ├── stratified/
│ │ ├── Kmean.py
│ │ └── GM.py
│ │
│ ├── classifiers/
│ │ ├── Knn.py
│ │ ├── AdaBoost.py
│ │ └── __init__.py
│ │
│ ├── regressors/
│ │ ├── __init__.py
│ │ └── XGBoost.py
│ │
│ └── filters/
│   ├── __init__.py
│   └── KM_filter.py
│
├── src/
│ └── main.py
│
├── utils/
│ ├── load_data.py
│ └── evaluation.py
│
├── requirements.txt
├── README.md
└── main.py # Example training & evaluation script
```

---

## 📦 Dataset Summary

| File | Description |
|------|--------------|
| `anchors_pos.csv` | Positions of the 11 sensors (`sensor_id, x, y`) |
| `centers_rssi.csv` | Training RSSI values for each anchor |
| `centers_pos_cell.csv` | Ground truth coordinates and cell ID for training samples |
| `tests_rssi.csv` | Test RSSI values |
| `tests_pos_cell.csv` | Ground truth coordinates and cell ID for test samples |
| `center_points_plot.png` | Distribution of training samples |
| `test_points_plot.png` | Distribution of test samples |

- **Training data:** 4,800 samples (10 samples per grid cell)
- **Test data:** 500 randomly generated points
- **RSSI values:** Computed using a simulated signal propagation model with noise

---

## ⚙️ Quick Start

### 1️⃣ Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt
```

### 2️⃣ Load Data

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
```

### 3️⃣ Train & Evaluate a Model

```python
# Example: Import models
from models.LightGBM import LightGBM
from models.linear_discriminant_analysis import linear_discriminant_analysis
from models.Logistic import Logistic
import numpy as np
import pandas as pd
from utils.evaluation import compute_distances_error
from sklearn.metrics import accuracy_score

# Initialize models
models = [LightGBM(),linear_discriminant_analysis(),Logistic()]
results = {'distance_error': dict(), "accuracy": dict(
), 'max_distance': dict(), 'min_distance': dict()}
for model in models:
    print(f'Fitting {model.name}-------------------------------')
    model.fit(X_train=X_train, df_y_train=df_y_train)
    y_pred_id, y_pred_coor = model.predict(X_test)
    accuracy = accuracy_score(y_test_id, y_pred_id)
    dis_error, max_dis, min_dis = compute_distances_error(y_test_coor, y_pred_coor)
    results['distance_error'][model.name] = dis_error
    results['accuracy'][model.name] = accuracy
    results['max_distance'][model.name] = max_dis
    results['min_distance'][model.name] = min_dis
```

## 📊 Example Benchmark

|Model| Mean Positioning Error (m) |cell prediction accuracy |max distance error|min distance error|
|-----| -------------------------- |-------------------------|------------------|------------------|
| AdaBoost      | 9.416312                   |0.004  | 21.936217  |  0.566265|
| LogisticRegression | 2.631670              | 0.188 | 19.637457| 0.116166|
