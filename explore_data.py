# from tsfresh import extract_features, extract_relevant_features, select_features
# from tsfresh.utilities.dataframe_functions import impute
# from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters
# import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pywt
from scipy import signal

# if __name__ == '__main__':

df_signal = pq.read_pandas('input/train.parquet', columns=["79"]).to_pandas()

sig = np.ravel(df_signal.iloc[:, 0].to_numpy())

t = df_signal.index.to_numpy()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
ax1.plot(t, sig)
ax1.set_title('Signal')
# ax1.axis([0, 1, -2, 2])
print(sig.shape)

sos = signal.butter(50, 65, 'hp', fs=1000, output='sos')
filtered = signal.sosfilt(sos, sig)
print(filtered.shape)

ax2.plot(t, filtered)
ax2.set_title('After 50 Hz high-pass filter')
# ax2.axis([0, 1, -2, 2])
ax2.set_xlabel('Time [seconds]')


def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


x = filtered
wavelet = 'haar'
level = 1
coeff = pywt.wavedec(x, wavelet, mode="per")
sigma = (1 / 0.6745) * maddest(coeff[-level])
uthresh = sigma * np.sqrt(2 * np.log(len(x)))
coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
x_dn = pywt.waverec(coeff, wavelet, mode='per')
print(x_dn.shape)

ax3.plot(t, x_dn)
ax3.set_title('After HP and Denoising')
# ax2.axis([0, 1, -2, 2])
ax2.set_xlabel('Time [seconds]')


def delete_repeat(a):
    b = np.empty(a.shape[0], dtype=a.dtype)

    for i in range(len(a) - 1):
        # print(i)

        if (a[i] == a[i - 1]) & (a[i] == a[i + 1]):
            b[i] = 99999
        else:
            b[i] = a[i]

    return b


x_deleted = delete_repeat(x_dn)
x_deleted_cond = (x_deleted < 99998)
x_deleted = x_deleted[x_deleted_cond]
print(x_deleted.shape)
t_deleted = t[x_deleted_cond]

ax4.plot(t_deleted, x_deleted)
ax4.set_title('After deleting')
# ax2.axis([0, 1, -2, 2])
ax4.set_xlabel('Time [seconds]')

plt.tight_layout()
plt.show()

# extraction_settings = EfficientFCParameters()
#
# master_df = pd.DataFrame({0: x_deleted,
#                           1: np.repeat(0, x_deleted.shape[0]),
#                           2: t_deleted})
#
# X = extract_features(master_df, column_id=1, column_sort=2, impute_function=impute, default_fc_parameters=extraction_settings)
#
# print(X.head(5))
