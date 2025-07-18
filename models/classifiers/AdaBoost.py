from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.evaluation import id_to_coor
import joblib


class AdaBoost:
    def __init__(self):
        """
        Initialize the AdaBoost classifier.

        The classifier uses scikit-learn's AdaBoost implementation with 100 estimators
        and a fixed random state for reproducibility.
        A LabelEncoder is used to encode and decode categorical cell IDs.
        """
        self.name = 'AdaBoost'
        self.model = AdaBoostClassifier(n_estimators=100, random_state=1)
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def fit(self, df_X_train:  pd.DataFrame, df_y_train: pd.DataFrame) -> None:
        """
        Train the AdaBoost classifier.

        Args:
            df_X_train (pandas.DataFrame): RSSI measurements for each sensor.
            df_y_train (pandas.DataFrame): A DataFrame containing the ground truth information.
                It must include the column 'cell_id' representing the class label for each point.

        Returns:
            None
        """
        X_train = self.scaler.fit_transform(df_X_train)
        y_train = self.encoder.fit_transform(df_y_train['cell_id'])
        self.model.fit(X_train, y_train)
        return

    def predict(self, df_X_test: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Predict cell IDs for the given test data and map them to coordinates.

        Args:
            X_test (pandas.DataFrame): RSSI measurements for each sensor.

        Returns:
            Tuple[numpy.ndarray, pandas.DataFrame]: A tuple containing:
                - A numpy array of predicted cell IDs.
                - A pandas DataFrame with the corresponding predicted coordinates
                  including columns 'x' and 'y'.
        """
        X_test = self.scaler.transform(df_X_test)
        y_pred_id = self.encoder.inverse_transform(self.model.predict(X_test))
        return y_pred_id, id_to_coor(y_pred_id)

    def save_parameters(self, path=f'./ckpts/'):
        print(f'saving model: {self.name}')
        joblib.dump(self.model, path+f'{self.name}_model.pkl')
        joblib.dump(self.scaler, path+f'{self.name}_scaler.pkl')
        joblib.dump(self.encoder, path+f'{self.name}_encoder.pkl')
        return

    def load_parameters(self, path=f'./ckpts/'):
        print(f'loading model: {self.name}')
        self.model = joblib.load(path+f'{self.name}_model.pkl')
        self.scaler = joblib.load(path+f'{self.name}_scaler.pkl')
        self.encoder = joblib.load(path+f'{self.name}_encoder.pkl')
        return
