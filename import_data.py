import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from scipy import signal
import pywt

if __name__ == '__main__':

    # <<<  Import Train Data >>> #

    # 800,000 data points taken over 20 ms
    # Grid operates at 50hz, 0.02 * 50 = 1, so 800k samples in 20 milliseconds will capture one complete cycle
    # Number of train signals 8712

    train_metadata = pd.read_csv('input/metadata_train.csv')  # read train data
    # print("Number of train signals: {}".format(train_metadata.shape[0]))

    start_train = 0  # number of columns to load
    end_train = 8712


    def maddest(d, axis=None):
        """
        Mean Absolute Deviation
        """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)


    def delete_repeat(a):
        b = np.empty(a.shape[0], dtype=a.dtype)

        for i in range(len(a) - 1):
            # print(i)

            if (a[i] == a[i - 1]) & (a[i] == a[i + 1]):
                b[i] = 99999
            else:
                b[i] = a[i]

        return b


    def high_pass_filter(sig):
        sos = signal.butter(50, 55, 'hp', fs=1000, output='sos')
        filtered = signal.sosfilt(sos, sig)
        return filtered


    def de_noising(x):
        wavelet = 'haar'
        level = 1
        coeff = pywt.wavedec(x, wavelet, mode="per")
        sigma = (1 / 0.6745) * maddest(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        x_dn = pywt.waverec(coeff, wavelet, mode='per')
        return x_dn


    # function to read each signal and convert into features
    def transform_ts(start, end, file):
        train_columns = pq.read_schema(file).names  # List with all column names to test
        # print(train_columns)
        X = pd.DataFrame(data=None)

        for i in train_columns[start:end]:
            df_signal = pq.read_pandas(file, columns=[i]).to_pandas()
            # turn parquet to dataframe of one single signal
            # print("Shape of signal data {}".format(df_signal.shape))

            sig = np.ravel(df_signal.iloc[:, 0].to_numpy())  # turn to numpy
            t = df_signal.index.to_numpy()  # turn time to numpy

            x_dn = de_noising(high_pass_filter(sig))
            x_deleted = delete_repeat(x_dn)
            x_deleted_cond = (x_deleted < 99998)
            x_deleted = x_deleted[x_deleted_cond]
            print(x_deleted.shape)
            t_deleted = t[x_deleted_cond]

            # Generating New Time Series Features from signal
            master_train = pd.DataFrame({0: x_deleted,
                                         1: np.repeat(i, x_deleted.shape[0]),
                                         2: t_deleted})
            # print("Shape of master train data {}".format(master_train.shape))
            # master_train.to_csv('output/master_train.csv')

            extraction_settings = EfficientFCParameters()
            X_signal = extract_features(master_train, column_id=1, column_sort=2, impute_function=impute,
                                        default_fc_parameters=extraction_settings)

            print("Number of extracted features in {}: {}.".format(i, X_signal.shape[1]))
            X = X.append(X_signal)

        return X


    X = transform_ts(start_train, end_train, 'input/train.parquet')  # run transform_ts function

    # Merge with metadata
    X = X.merge(train_metadata, left_on="id", right_on="signal_id")

    y = X["target"]
    print("Shape of Y {}".format(X.shape))
    y.to_csv("input/y.csv", index=False)

    X = X.drop(["signal_id", "target"], axis=1)
    print("Shape of X {}".format(X.shape))
    X.to_csv("input/X.csv", index=False)

    # <<< Import Test Data >>> #

    test_metadata = pd.read_csv('input/metadata_test.csv')  # read train data
    # print(len(test_metadata))
    start_test = 0  # test_metadata.shape[0]
    end_test = 20337
    X_test = transform_ts(start_test, end_test, 'input/test.parquet')  # run transform_ts function

    # Merge with metadata
    X_test = X_test.merge(test_metadata, left_on="id", right_on="signal_id")

    X_test = X_test.drop(["signal_id"], axis=1)
    print("Shape of X_test {}".format(X_test.shape))
    X_test.to_csv("input/X_test.csv", index=False)
