import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


file = "kanal1/move5.csv"
data = remove_bad_carriers(amp_calc(parse(file)))

plt.plot(data)
plt.title(file)
plt.show()
