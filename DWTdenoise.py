import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ML
#brugt https://www.kaggle.com/code/theoviel/denoising-with-direct-wavelet-transform/notebook til at forstå hvordan man gør i python
# extract og vælg subarrier og wavelet form.
#bruger db4 based på https://ieeexplore-ieee-org.proxy.findit.cvt.dk/stamp/stamp.jsp?tp=&arnumber=9217780
#subcarrier = 14




csidata = ML.parse("kanal10.csv")
amp = ML.amp_calc(csidata)
df = pd.DataFrame(amp)
filtered=[]
wavelet="db4"


def filter(df):

    for subcarrier in range(192):
        x = df.iloc[:, subcarrier]
        coef = pywt.wavedec(x, wavelet=wavelet, mode="per")
        #print(len(coef))
        mad = np.mean(np.absolute(coef[-1] - np.mean(coef[-1], axis=None)), axis=None)
        #N0.75 procentile= cirka 0.6745
        sigma = (1 / 0.6745) * mad
        thresh = sigma * np.sqrt(2 * np.log(len(x)))
        coef[1:] = (pywt.threshold(i, value=thresh, mode='hard') for i in coef[1:])
        filter=pywt.waverec(coef, wavelet=wavelet, mode='per')
        filter = filter.tolist()
        filtered.append(filter)
    return filtered


# #set hvilken subcarrier der skal pritntes
# subcarrier=130
# for subcarrier in range(134,192):
#     plt.figure(figsize=(10, 6))
#     plt.plot(df.iloc[:][subcarrier], label='Raw')
#     plt.plot(filtered[:][subcarrier], label='Filtered')
#     plt.legend()
#     plt.title(f"DWT Denoising with db4 Wavelet Subcarrier {subcarrier}", size=15)
#     plt.show(block=False)
#     plt.pause(1111.5)
#     plt.close()
#
#     # sigma=(1/0.6745)*coef.mad(axis=1)
