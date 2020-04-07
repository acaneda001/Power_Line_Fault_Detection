import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, matthews_corrcoef, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from scipy import signal
from xgboost import XGBRFClassifier
import xgboost as xgb
import optuna
import functools

import numpy as np

if __name__ == '__main__':

    # Import Data

    X_test = pd.read_csv("input/X_test.csv")
    print(X_test.shape)
    X = pd.read_csv("input/X.csv")
    print(X.shape)
    y = pd.read_csv("input/y.csv", squeeze=True)
    print(y.shape)

    train_metadata = pd.read_csv('input/metadata_train.csv')  # read train data
    test_metadata = pd.read_csv('input/metadata_test.csv')  # read train data


    def ML_model(X_train, X_valid, y_train, y_valid):
        my_model = XGBRFClassifier(random_state=1, scale_pos_weight=17)
        my_model.fit(X_train, y_train)
        predictions = pd.Series(my_model.predict(X_valid))

        if not y_valid.empty:
            print("Mean Absolute Error: " + str(mean_absolute_error(y_valid, predictions)))
            print("Matthews correlation coefficient: " + str(matthews_corrcoef(y_valid, predictions)))

        return predictions


    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, stratify=y,
                                                          random_state=1)  # split X, y into 80% train 20% valid

    my_imputer = SimpleImputer(strategy="median")
    my_imputer.fit(X_train)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = pd.DataFrame(scaler.transform(my_imputer.transform(X_train)))
    X_valid = pd.DataFrame(scaler.transform(my_imputer.transform(X_valid)))

    # -----------------------------------
    prediction = ML_model(X_train, X_valid, y_train, y_valid)

    # <<< Select relevant features >>> #

    # y_train.reset_index(drop=True, inplace=True)
    #
    # print(X_train.shape)
    # print(y_train.shape)
    #
    # X_train_filtered = select_features(X_train, y_train)
    # print("Number of relevant features {}/{}".format(X_train_filtered.shape[1], X_train.shape[1]))
    #
    X_test.columns = X_valid.columns  # rename X_test columns to X_valid column names
    #
    # X_valid = X_valid[X_train_filtered.columns]
    #
    # X_test = X_test[X_train_filtered.columns]
    #
    # prediction_filtered = ML_model(X_train_filtered, X_valid, y_train, y_valid)

    test_prediction = ML_model(X_train, X_test, y_train, pd.DataFrame(data=None))

    output = pd.DataFrame({'signal_id': test_metadata["signal_id"],
                       'target': test_prediction})

    output.to_csv('output/submission_2.csv', index=False)

    # <<< Hyper parameter optimisation >>> #

    # best_params = {'n_estimators': 949, 'max_depth': 18, 'min_child_weight': 4,
    #                'scale_pos_weight': 10, 'subsample': 0.7, 'colsample_bytree': 0.5}
    #
    # clf = xgb.XGBClassifier(**best_params)
    # clf.fit(X_train, y_train)
    # predictions_hyper_par = pd.Series(clf.predict(X_valid))
    #
    # print("Mean Absolute Error: " + str(mean_absolute_error(y_valid, predictions_hyper_par)))
    # print("Matthews correlation coefficient: " + str(matthews_corrcoef(y_valid, predictions_hyper_par)))
    #
    # # <<< To improve accuracy, create a new  model which you will train on all training data >>> #
    #
    #
    #
    # test_prediction = pd.Series(clf.predict(X_test))
    #
    # output = pd.DataFrame({'signal_id': test_metadata["signal_id"],
    #                        'target': test_prediction})
    #
    # output.to_csv('output/submission.csv', index=False)
