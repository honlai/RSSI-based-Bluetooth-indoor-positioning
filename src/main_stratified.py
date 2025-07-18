import pandas as pd
from utils.load_data import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from models.classifiers.AdaBoost import AdaBoost
# from methods.bagging import BaggingWrapper
# from methods.decision_tree import decision_tree
# from methods.gradient_boosting import gradient_boosting_classifier
# from methods.histogram_based_gradient_boosting import HBGC
# from methods.LightGBM import LightGBM
# from methods.linear_discriminant_analysis import linear_discriminant_analysis
from models.classifiers.Logistic import Logistic
# from methods.multi_layer_perceptron import MLPC
# from methods.NuSVC import nu_support_vector_classifier
# from methods.quadratic_discriminant_analysis import quadratic_discriminant_analysis
# from methods.random_forest import random_forest_classifier
# from methods.stochastic_gradient_descent import stochastic_gradient_descent
# from methods.SVM import support_vector_classifier
# from methods.XGBoostTuning import xgboost_classifier_bayessearch
# from methods.XGBoost import xgboost_classifier, xgboost_regressor
# from methods.MLP import MLP_classifier
from utils.evaluation import compute_distances_error, draw_distribution_of_dist_err
from models.stratified.KMeans import KMeans
from models.stratified.GaussianMixture import GaussianMixture
from models.stratified.MeanShift import MeanShift
if __name__ == '__main__':
    # data processing
    print("Data processing...")
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

    # fitting model
    k = 10
    stratified = [MeanShift, KMeans, GaussianMixture]
    models = [AdaBoost, Logistic]
    stratified_model = [stratify(model, k)
                        for model in models for stratify in stratified]
    results = {'distance_error': dict(), "accuracy": dict(
    ), 'max_distance': dict(), 'min_distance': dict(), 'q_25': dict(), 'q_50': dict(), 'q_75': dict()}
    for model in stratified_model:
        print(f'Fitting {model.name}-------------------------------')
        try:
            model.fit(X_train=X_train, df_y_train=df_y_train)
            y_pred_id, y_pred_coor = model.predict(X_test)
        except:
            print(f'{model.name} fail')
            continue
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
    print('Done---------------------------------------------------')
    print('results:')
    print(pd.DataFrame(results))
