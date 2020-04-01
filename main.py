import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, matthews_corrcoef
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from scipy import signal
from xgboost import XGBRFClassifier

# Import Data
X = pd.read_csv("output/X.csv")
X_test = pd.read_csv("output/X_test.csv")
y = pd.read_csv("output/y.csv")
train_metadata = pd.read_csv('input/metadata_train.csv')  # read train data
test_metadata = pd.read_csv('input/metadata_test.csv')  # read train data


def xgboost_model(X_train, X_valid, y_train, y_valid):

    my_model = XGBRFClassifier(n_estimators=1000, learning_rate=0.05, random_state=1)
    my_model.fit(X_train, y_train)
    predictions = my_model.predict(X_valid)
    print(predictions)

    if not X_valid.empty:
        print("Mean Absolute Error: " + str(mean_absolute_error(y_valid, predictions)))
        print("Matthews correlation coefficient: " + str(matthews_corrcoef(y_valid, predictions)))

    return predictions


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2)  # split X, y into 80% train 20% valid

prediction = xgboost_model(X_train, X_valid, y_train, y_valid)

# Select relevant features
X_train_filtered = select_features(X_train, y_train.to_numpy())
print("Number of relevant features {}/{}".format(X_train_filtered.shape[1], X_train.shape[1]))

prediction_filtered = xgboost_model(X_train_filtered, X_valid, y_train, y_valid)

# To improve accuracy, create a new  model which you will train on all training data

test_prediction = xgboost_model(X_test[[X_train_filtered.columns]], None, y, None)

output = pd.DataFrame({'signal_id': test_metadata["signal_id"],
                       'target': test_prediction})

output.to_csv('output/submission.csv', index=False)



