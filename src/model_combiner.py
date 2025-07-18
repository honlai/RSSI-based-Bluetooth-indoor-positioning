from models.filters.KM_filters import (
    KalmanFilterSet,
    MultiKalmanFilter,
    KalmanWithVelocitySet,
    ExtendedKalmanFilterSet,
    ExtendedKalmanFilterSetNonlinear
)
from models.classifiers.AdaBoost import AdaBoost
# from models.decision_tree import decision_tree
# from models.gradient_boosting import gradient_boosting_classifier
# from models.histogram_based_gradient_boosting import HBGC
# from models.LightGBM import LightGBM
# from models.linear_discriminant_analysis import linear_discriminant_analysis
from models.classifiers.Logistic import Logistic
# from models.multi_layer_perceptron import MLPC
# from models.NuSVC import nu_support_vector_classifier
# from models.quadratic_discriminant_analysis import quadratic_discriminant_analysis
# from models.random_forest import random_forest_classifier
# from models.stochastic_gradient_descent import stochastic_gradient_descent
# from models.SVM import support_vector_classifier
# from models.XGBoostTuning import xgboost_classifier_bayessearch

# from models.XGBoost import xgboost_classifier, xgboost_regressor
# from models.MLP import MLP_classifier
import pandas as pd
import numpy as np
from utils.load_data import load_data
from utils.evaluation import compute_distances_error, draw_distribution_of_dist_err
from sklearn.metrics import accuracy_score


class ModelCombiner:
    def __init__(self, initial_values: pd.Series, filter_name: str, model_name: str, PL_A=-59, PL_n=2.7):
        self.set_filter = {
            "kalman_filter_set": KalmanFilterSet,
            "multi_kalman_filter": MultiKalmanFilter,
            "kalman_with_velocity_set": KalmanWithVelocitySet,
            "extended_kalman_filter_set": ExtendedKalmanFilterSet,
            "extended_kalman_filter_set_nonlinear": ExtendedKalmanFilterSetNonlinear
        }
        self.filter = self.set_filter.get(
            filter_name, lambda: None)(initial_values)
        set_model = {
            "AdaBoost": AdaBoost,
            "Logistic": Logistic
        }
        self.model = set_model.get(model_name, lambda: None)()
        # self.model.load_parameters()
        self.name = self.filter.name+self.model.name

    def fit(self, df_X_train, df_y_train):
        training_filter = self.set_filter.get(
            filter_name, lambda: None)(df_X_train.iloc[0])
        for i, row in df_X_train.iterrows():
            df_X_train.iloc[i] = training_filter.update(row)
        self.model.fit(df_X_train, df_y_train)
        return

    def update(self, new_row: pd.Series) -> pd.DataFrame:
        filtered = [self.filter.update(new_row)]
        df_result_t = pd.DataFrame(filtered)
        y_pred_id, y_pred_coor = self.model.predict(df_result_t)
        return y_pred_coor


def result_combine(result):
    ids = []
    xs = []
    ys = []
    for item in result:
        ids.append(item['cell'][0])
        xs.append(item['x'][0])
        ys.append(item['y'][0])

    return np.array(ids), pd.DataFrame({'x': xs, 'y': ys, 'cell': ids})


if __name__ == '__main__':
    df_X_train, df_y_train, df_X_test, df_y_test = load_data(
        test_type="trajectory")
    y_test_coor = df_y_test[['x', 'y']]
    y_test_id = df_y_test['cell_id']
    df_data = df_X_test
    list_filter_name = [
        "kalman_filter_set",
        "multi_kalman_filter",
        "kalman_with_velocity_set",
        "extended_kalman_filter_set",
        "extended_kalman_filter_set_nonlinear"
    ]
    model_names = [
        "AdaBoost",
        "Logistic"
    ]
    results = {'distance_error': dict(), "accuracy": dict(
    ), 'max_distance': dict(), 'min_distance': dict(), 'q_25': dict(), 'q_50': dict(), 'q_75': dict()}
    for model_name in model_names:
        for filter_name in list_filter_name:

            mb1 = ModelCombiner(
                initial_values=df_data.iloc[0],
                filter_name=filter_name,
                model_name=model_name,
                PL_A=-59,
                PL_n=2.7,)
            print(f'Fitting {mb1.name}-------------------------------')
            mb1.fit(df_X_train=df_X_train, df_y_train=df_y_train)

            result_list = []
            for _, row in df_data.iterrows():
                y_pred = mb1.update(row)
                result_list.append(y_pred)
            y_pred_id, y_pred_coor = result_combine(result_list)
            print(y_pred_coor)

            accuracy = accuracy_score(y_test_id, y_pred_id)
            dis_error, max_dis, min_dis, q25, q50, q75 = compute_distances_error(
                y_test_coor, y_pred_coor)
            results['distance_error'][mb1.name] = dis_error
            results['accuracy'][mb1.name] = accuracy
            results['max_distance'][mb1.name] = max_dis
            results['min_distance'][mb1.name] = min_dis
            results['q_25'][mb1.name] = q25
            results['q_50'][mb1.name] = q50
            results['q_75'][mb1.name] = q75
            draw_distribution_of_dist_err(y_test_coor, y_pred_coor, mb1.name)

    print('Done---------------------------------------------------')
    print('results:')
    print(pd.DataFrame(results))
