import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, matthews_corrcoef, accuracy_score

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import optuna
import functools

if __name__ == '__main__':
    # Import Data

    X = pd.read_csv("input/X.csv")
    print(X.shape)
    y = pd.read_csv("input/y.csv", squeeze=True)
    print(y.shape)

    train_metadata = pd.read_csv('input/metadata_train.csv')  # read train data
    test_metadata = pd.read_csv('input/metadata_test.csv')  # read train data

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, stratify=y,
                                                          random_state=1)  # split X, y into 80% train 20% valid

    my_imputer = SimpleImputer(strategy="median")
    my_imputer.fit(X_train)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = pd.DataFrame(scaler.transform(my_imputer.transform(X_train)))
    X_valid = pd.DataFrame(scaler.transform(my_imputer.transform(X_valid)))


    def opt(X_train, y_train, X_test, y_test, trial):
        # param_list
        n_estimators = trial.suggest_int('n_estimators', 0, 1000)
        max_depth = trial.suggest_int('max_depth', 1, 20)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
        # learning_rate = trial.suggest_discrete_uniform('learning_rate', 0.01, 0.1, 0.01)
        scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 100)
        subsample = trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1)
        colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1)

        xgboost_tuna = xgb.XGBClassifier(
            random_state=42,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            # learning_rate = learning_rate,
            scale_pos_weight=scale_pos_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )
        xgboost_tuna.fit(X_train, y_train)
        tuna_pred_test = xgboost_tuna.predict(X_test)

        return 1.0 - (accuracy_score(y_test, tuna_pred_test))


    study = optuna.create_study()
    study.optimize(functools.partial(opt, X_train, y_train, X_valid, y_valid), n_trials=100)
    print(study.best_trial)
    print(study.best_params)
