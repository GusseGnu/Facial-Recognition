import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pywt

matplotlib.use('TkAgg')

def parse(path):
    csi_df = pd.read_csv(path)
    print(csi_df.shape)
    csi_df.drop(csi_df.loc[csi_df['sig_mode'] == 0].index, inplace=True)
    print(csi_df.shape)
    CSI_col = csi_df['CSI_DATA'].copy()
    print(CSI_col.shape)
    final_csi_data = np.array([np.fromstring(csi_datum.strip('[ ]'), dtype=int, sep=' ') for csi_datum in CSI_col])
    print("csi data shape", final_csi_data.shape)
    return final_csi_data


def amp_calc(csi_data):
    amplitude = np.array([np.sqrt(data[::2]**2 + data[1::2]**2) for data in csi_data])
    return amplitude


def remove_bad_carriers(amp_data):
    good_ones = amp_data
    n = 0
    for i in range(0, amp_data.shape[1]):
        count = 0
        for j in range(1, amp_data.shape[0]):
            if amp_data[j][i] == amp_data[j-1][i]:
                count += 1
            else:
                count = 0
            if count > 10:
                good_ones = np.delete(good_ones, i-n, 1)
                n += 1
                print("Deleted " + str(i+1))
                break
    return good_ones


def denoise_data(df, value):
    print("Filtering received shape", df.shape)
    df = pd.DataFrame(df)
    filtered = []
    wavelet = "db4"
    for subcarrier in range(df.shape[1]):
        x = df.iloc[:, subcarrier]
        coef = pywt.wavedec(x, wavelet=wavelet, mode="per")
        mad = np.mean(np.absolute(coef[-1] - np.mean(coef[-1], axis=None)), axis=None)
        # N0.75 procentile= cirka 0.6745
        sigma = (1 / value) * mad
        thresh = sigma * np.sqrt(2 * np.log(len(x)))
        coef[1:] = (pywt.threshold(i, value=thresh, mode='hard') for i in coef[1:])
        filter = pywt.waverec(coef, wavelet=wavelet, mode='per')
        filter = filter.tolist()
        filtered.append(filter)
    filtered = np.transpose(np.array(filtered))
    # filtered = np.delete(filtered, len(filtered)-1, axis=0)
    print("Filtering returning shape", filtered.shape)
    return filtered


# plt.subplot(2, 1, 1)
#
# file = "Through_wall/beggerumstilhed92.csv"
# data = remove_bad_carriers(amp_calc(parse(file)))
# # data = denoise_data(data, 0.7754)
# # plt.axvline(x=100, linestyle='dotted')
# plt.plot(data)
# plt.title(file)
#
# plt.subplot(2, 1, 2)
#
# file = "Through_wall/beggerumbevæg9.csv"
# data = remove_bad_carriers(amp_calc(parse(file)))
# # data = denoise_data(data, 0.7754)
#
# plt.plot(data)
# plt.title(file)
# plt.show()

file = "Bib_kanal_test/bib_kanal9_move.csv"
data = remove_bad_carriers(amp_calc(parse(file)))
data = np.array(data[:, range(data.shape[1]-50, data.shape[1])])
data = denoise_data(data, 0.7754)
plt.plot(data)
plt.title("Line graph")
plt.ylabel("Amplitude (dB)")
plt.xlabel("Packet number")
plt.show()
