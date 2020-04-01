import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
from scipy import signal


if __name__ == '__main__':

    #  Import Train Data

    # 800,000 data points taken over 20 ms
    # Grid operates at 50hz, 0.02 * 50 = 1, so 800k samples in 20 milliseconds will capture one complete cycle
    # Number of train signals 8712

    train_metadata = pd.read_csv('input/metadata_train.csv')  # read train data
    # print("Number of train signals: {}".format(train_metadata.shape[0]))

    start_train = 5  # number of columns to load
    end_train = 6
    target_freq = 0.5  # Hz  -> resample to reduce the size of array

    # simple function to resample signals
    def resample_signal(df, freq_factor):
        df_re = pd.DataFrame(data=None)
        samples = len(df)  # number of samples per signal = 800,000
        columns = len(df.columns)  # number of signals to resample simultaneously

        # print(samples)
        for i in range(columns):
            df_re[i] = signal.resample(df.iloc[:, i], samples // freq_factor)

        # Plotting
        # plt.plot(df.index, df.iloc[:, 0], 'b.',
        #          np.linspace(0, len(df), len(df_re), endpoint=False), df_re.iloc[:, 0], 'r.')
        # plt.show()
        return df_re

    # function to read each signal and convert into features
    def transform_ts(start, end, target_freq, file):
        train_columns = pq.read_schema(file).names   # List with all column names to test
        # print(train_columns)
        X = pd.DataFrame(data=None)

        for i in train_columns[start:end]:
            df_signal = pq.read_pandas(file, columns=[i]).to_pandas()
            # turn parquet to dataframe of one single signal
            # print("Shape of signal data {}".format(df_signal.shape))

            freq_factor = int(50 / target_freq)  # factor to resample length of signal
            df_signal_re = resample_signal(df_signal, freq_factor)  # resample signal

            # print("Shape of resampled train data {}".format(df_signal_re.shape))

            # Generating New Time Series Features from signal
            master_train = pd.DataFrame({0: df_signal_re.iloc[:, 0].values.flatten(),
                                         1: np.repeat(int(i), df_signal_re.shape[0])})

            # print("Shape of master train data {}".format(master_train.shape))
            # master_train.to_csv('output/master_train.csv')

            extraction_settings = EfficientFCParameters()
            X_signal = extract_features(master_train, column_id=1, impute_function=impute,
                                        default_fc_parameters=extraction_settings)

            print("Number of extracted features in {}: {}.".format(i, X_signal.shape[1]))
            X = X.append(X_signal)

        print("Shape of the resampled train data {}/{}".format(df_signal_re.shape[0], df_signal.shape[0]))

        return X

    X = transform_ts(start_train, end_train, target_freq, 'input/train.parquet')  # run transform_ts function

    # Merge with metadata
    X = X.merge(train_metadata, left_on="id", right_on="signal_id")

    y = X[["target"]]
    print("Shape of Y {}".format(X.shape))
    y.to_csv("output/y.csv")

    X = X.drop(["signal_id", "target"], axis=1)
    print("Shape of X {}".format(X.shape))
    X.to_csv("output/X.csv")

    # Import Test Data

    test_metadata = pd.read_csv('input/metadata_test.csv')  # read train data
    start_test = 0  # test_metadata.shape[0]
    end_test = 5
    X_test = transform_ts(start_test, end_test, target_freq, 'input/test.parquet')  # run transform_ts function

    # Merge with metadata
    X_test = X_test.merge(test_metadata, left_on="id", right_on="signal_id")

    X_test = X_test.drop(["signal_id"], axis=1)
    print("Shape of X_test {}".format(X_test.shape))
    X_test.to_csv("output/X_test.csv")
