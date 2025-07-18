import pandas as pd
from utils.load_data import load_data
from utils.evaluation import compute_distances_error, draw_distribution_of_dist_err
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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


if __name__ == '__main__':
    # data processing
    print("Data processing...")
    df_X_train, df_y_train, df_X_test, df_y_test = load_data(
        test_type='trajectory')
    y_test_coor = df_y_test[['x', 'y']]
    y_test_id = df_y_test['cell_id']

    # fitting model
    models = [AdaBoost(), Logistic()]
    results = {'distance_error': dict(), "accuracy": dict(
    ), 'max_distance': dict(), 'min_distance': dict(), 'q_25': dict(), 'q_50': dict(), 'q_75': dict()}
    for model in models:
        print(f'Fitting {model.name}-------------------------------')
        model.fit(df_X_train=df_X_train, df_y_train=df_y_train)
        y_pred_id, y_pred_coor = model.predict(df_X_test)
        accuracy = accuracy_score(y_test_id, y_pred_id)
        dis_error, max_dis, min_dis, q25, q50, q75 = compute_distances_error(
            y_test_coor, y_pred_coor)
        results['distance_error'][model.name] = dis_error
        results['accuracy'][model.name] = accuracy
        results['max_distance'][model.name] = max_dis
        results['min_distance'][model.name] = min_dis
        results['q_25'][model.name] = q25
        results['q_50'][model.name] = q50
        results['q_75'][model.name] = q75
        draw_distribution_of_dist_err(y_test_coor, y_pred_coor, model.name)
        model.save_parameters()
    print('Done---------------------------------------------------')
    print('results:')
    print(pd.DataFrame(results))
