import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as KM


class KMeans():
    def __init__(self, model, k):
        self.model = model
        self.k = k
        self.models = [model() for _ in range(k)]
        self.kmeans = KM(n_clusters=k, random_state=42, n_init='auto')
        self.name = f'KMeans_stratified_{model().name}'

    def fit_stratified(self, X):
        z = self.kmeans.fit_predict(X)
        return z

    def predict_stratified(self, X):
        z = self.kmeans.predict(X)
        return z

    def fit(self, X_train, df_y_train):
        z = self.fit_stratified(X_train)
        for i in range(self.k):
            mask = z == i
            self.models[i].fit(X_train[mask], df_y_train[mask])
        return

    def predict(self, X_test):
        z = self.predict_stratified(X_test)
        batch_id_results = []
        batch_coor_results = []
        for i in range(self.k):
            filtered_indices = np.where(z == i)[0]
            y_pred_id, y_pred_coor = self.models[i].predict(X_test[z == i])
            y_pred_id = pd.DataFrame(y_pred_id)
            y_pred_id.index = filtered_indices
            y_pred_coor.index = filtered_indices
            batch_id_results.append(y_pred_id)
            batch_coor_results.append(y_pred_coor)
        return pd.concat(batch_id_results).sort_index().values, pd.concat(batch_coor_results).sort_index()
