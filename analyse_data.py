import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, matthews_corrcoef, accuracy_score, recall_score, roc_curve, auc, \
    precision_score
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from scipy import signal
from xgboost import XGBRFClassifier
import xgboost as xgb
import optuna
import functools
import eli5
from eli5.sklearn import PermutationImportance
import shap

import numpy as np

# Import Data
X_test = pd.read_csv("input/X_test.csv")
X = pd.read_csv("input/X.csv")
y = pd.read_csv("input/y.csv", squeeze=True)
X_test_filtered = pd.DataFrame(X_test).iloc[:, [173, 141, 530, 683, 661, 498, 48, 183, 206, 716, 697, 185, 211, 624, 671, 623, 67, 111, 118, 129]]

# train_metadata = pd.read_csv('input/metadata_train.csv')  # read train data
test_metadata = pd.read_csv('input/metadata_test.csv')  # read train data

def ROC_curve(y_test, y_score):
    # Compute ROC curve and ROC area for each class

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1, test_size=0.35)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# # combine them back for resampling
# train_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
#
# # separate minority and majority classes
# negative = train_data[train_data.target == 0]
# positive = train_data[train_data.target == 1]
#
# # downsample majority
# neg_downsampled = resample(negative,
#                            replace=True,  # sample with replacement
#                            n_samples=len(positive),  # match number in minority class
#                            random_state=27)  # reproducible results
# # combine minority and downsampled majority
# downsampled = pd.concat([positive, neg_downsampled]).dropna()
# check new class counts
#
# X_train = pd.DataFrame(downsampled.drop(columns="target"), index=downsampled.index)
# y_train = pd.Series(downsampled["target"], index=downsampled.index)

my_model = XGBRFClassifier(random_state=1).fit(X_train, y_train)

predictions = my_model.predict(X_val)

print("Matthews Correlation Coefficient: " + str(matthews_corrcoef(predictions, y_val)))
print("Precision Score: " + str(precision_score(predictions, y_val)))
print("Recall Score: " + str(recall_score(predictions, y_val)))
ROC_curve(y_val, predictions)

X_train_filtered = pd.DataFrame(X_train).iloc[:, [173, 141, 530, 683, 661, 498, 48, 183, 206, 716, 697, 185, 211, 624, 671, 623, 67, 111, 118, 129]]
X_val_filtered = pd.DataFrame(X_val).iloc[:, [173, 141, 530, 683, 661, 498, 48, 183, 206, 716, 697, 185, 211, 624, 671, 623, 67, 111, 118, 129]]

my_model_filtered = XGBRFClassifier(**{'n_estimators': 610, 'max_depth': 14, 'min_child_weight': 10, 'scale_pos_weight': 84, 'subsample': 0.8, 'colsample_bytree': 0.8}).fit(X_train_filtered, y_train)

predictions_filtered = my_model_filtered.predict(X_val_filtered)

print("Matthews Correlation Coefficient: " + str(matthews_corrcoef(predictions_filtered, y_val)))
print("Precision Score: " + str(precision_score(predictions_filtered, y_val)))
print("Recall Score: " + str(recall_score(predictions_filtered, y_val)))
ROC_curve(y_val, predictions_filtered)

X_test_filtered.columns = X_train_filtered.columns

test_prediction = my_model_filtered.predict(X_test_filtered)

# prediction = PermutationImportance(my_model, random_state=1).fit(X_val, y_val)
#
# df_eli = eli5.format_as_dataframe(eli5.explain_weights(prediction))  # , feature_names=X_val.columns.tolist()
# df_eli.to_csv("output/eli.csv")
#
# # Create object that can calculate shap values
# explainer = shap.TreeExplainer(my_model)
#
# # calculate shap values. This is what we will plot.
# # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
# shap_values = explainer.shap_values(X_val)
#
# # Make plot. Index of [1] is explained in text below.
# shap.summary_plot(shap_values[1], X_val)
#
output = pd.DataFrame({'signal_id': test_metadata["signal_id"],
                       'target': test_prediction})
output.to_csv('output/submission_3.csv', index=False)
