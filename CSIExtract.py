import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def Parse(Path):
    csi_df = pd.read_csv(Path)
    csi_df.drop(csi_df.loc[csi_df['sig_mode'] == 0].index, inplace=True)
    CSI_Col = csi_df['CSI_DATA'].copy()
    csi_data = np.array([np.fromstring(csi_datum.strip('[ ]'), dtype=int, sep=' ') for csi_datum in CSI_Col])
    return csi_data


def AmplitudeCalc(csi_data):
  #  print(csi_data.shape)
        #amplitude = np.array([np.sqrt(i[::2] ** 2 + i[1::2] ** 2)])
    amplitude = np.array([np.sqrt(data[::2]**2 + data[1::2]**2) for data in csi_data])

    return amplitude

def PhaseCalc(csi_data):
    for i in csi_data:
        phase = np.array([np.arctan2(i[::2], i[1::2])])
        #phase = arctan(Im / Re)
    return phase
def removeNull(data):
    NULL_SUBCARRIERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 64, 65, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,128,129, 130, 131, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261,262,263, 264, 265, 266, 267, 382, 383]
    a_del = np.delete(data, NULL_SUBCARRIERS, 1)
    return a_del