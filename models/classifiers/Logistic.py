from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils.evaluation import id_to_coor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


class Logistic:
    def __init__(self):
        """
        Initialize the Logistic Regression model using scikit-learn.
        The model is configured for multinomial classification with the L-BFGS solver
        and a maximum of 1000 iterations to ensure convergence.
        """
        self.name = 'Logistic Regression'
        self.model = LogisticRegression(
            multi_class='multinomial', solver='lbfgs', max_iter=1000)
        self.scaler = StandardScaler()

    def fit(self, df_X_train:  pd.DataFrame, df_y_train: pd.DataFrame) -> None:
        """
        Train the logistic regression model.

        Args:
            X_train (pandas.DataFrame): RSSI measurements for each sensor.
            df_y_train (pandas.DataFrame): A DataFrame containing the ground truth information.
                It must include the columns 'x', 'y', and 'cell_id', where:
                    - 'x' and 'y' represent the coordinates,
                    - 'cell_id' represents the class label for each point.

        Returns:
            None
        """
        X_train = self.scaler.fit_transform(df_X_train)
        self.model.fit(X_train, df_y_train['cell_id'])
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
        y_pred_id = self.model.predict(X_test)
        return y_pred_id, id_to_coor(y_pred_id)

    def save_parameters(self, path=f'./ckpts/'):
        print(f'saving model: {self.name}')
        joblib.dump(self.model, path+f'{self.name}_model.pkl')
        joblib.dump(self.scaler, path+f'{self.name}_scaler.pkl')
        return

    def load_parameters(self, path=f'./ckpts/'):
        print(f'loading model: {self.name}')
        self.model = joblib.load(path+f'{self.name}_model.pkl')
        self.scaler = joblib.load(path+f'{self.name}_scaler.pkl')
        return
