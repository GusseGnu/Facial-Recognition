import matplotlib
import matplotlib.pyplot as plt
import pywt
import cv2
from PIL import Image
import io
import time
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')
#
# plt.rcParams["figure.autolayout"] = True
# plt.rcParams["figure.figsize"] = (3.2, 2.4)  # Number of hundreds of pixels as float
# while True:
#     plt.plot([1, 2, 3, 4], color='r')
#     plt.ylabel('some numbers')
#     start = time.process_time()
#     img_buff = io.BytesIO()
#     plt.savefig(img_buff, format='png')
#
#     im = Image.open(img_buff)
#     data = np.asarray(im)
#     # im.show(title="Image test")
#     cv2.imshow("TEST", data)
#     if cv2.waitKey(1) == ord('q'):  # q to quit
#         break
#     end = time.process_time()
#     print("Done in " + str(end-start) + " seconds")
#


def parse(path):
    csi_df = pd.read_csv(path)
    print(csi_df.shape)
    csi_df.drop(csi_df.loc[csi_df['sig_mode'] == 0].index, inplace=True)
    print(csi_df.shape)
    CSI_col = csi_df['CSI_DATA'].copy()
    print("CSI column shape", CSI_col.shape)
    temp = [np.fromstring(csi_datum.strip('[ ]'), dtype=int, sep=' ') for csi_datum in CSI_col]
    work_pls = []
    for packet in temp:
        if len(str(packet).split(" ")) > 600: # Random ass check
            work_pls.append(packet)
    final_csi_data = np.array(work_pls)
    print("Type is", final_csi_data[0][0].dtype)
    print("csi data shape", final_csi_data.shape)
    return final_csi_data


def remove_bad_carriers(amp_data):
    good_ones = amp_data
    n = 0
    for i in range(0, amp_data.shape[1]):
        count = 0
        for j in range(1, amp_data.shape[0]):
            if amp_data[j][i] == amp_data[j-1][i]:
                count += 1
            # if (amp_data[j - 1][i] * 1.05) > amp_data[j][i] > (amp_data[j - 1][i] * 0.955):
            #     count += 1
            else:
                count = 0
            if count > 5:
                good_ones = np.delete(good_ones, i-n, 1)
                n += 1
                print("Deleted " + str(i+1))
                break
    return good_ones

plt.rcParams["figure.figsize"] = [10.80, 7.20]
plt.rcParams["figure.autolayout"] = True


def amp_calc(csi_data):
    amplitude = np.array([np.sqrt(data[::2]**2 + data[1::2]**2) for data in csi_data])
    return amplitude


def filter_data(df):
    df = pd.DataFrame(df)
    print(df.shape)
    filtered = []
    wavelet = "db4"
    for subcarrier in range(df.shape[1]):
        x = df.iloc[:, subcarrier]
        coef = pywt.wavedec(x, wavelet=wavelet, mode="per")
        mad = np.mean(np.absolute(coef[-1] - np.mean(coef[-1], axis=None)), axis=None)
        #N0.75 procentile= cirka 0.6745
        sigma = (1 / 0.7754) * mad
        thresh = sigma * np.sqrt(2 * np.log(len(x)))
        coef[1:] = (pywt.threshold(i, value=thresh, mode='hard') for i in coef[1:])
        filter=pywt.waverec(coef, wavelet=wavelet, mode='per')
        filter = filter.tolist()
        filtered.append(filter)
    return np.transpose(np.array(filtered))


def four_plots():
    shape = df.shape
    print(shape)

    plot1 = df[0:int(shape[0]/2), 0:int(shape[1]/2)]
    print(plot1.shape)
    plt.plot(plot1)
    plt.show()

    plot2 = df[0:int(shape[0]/2), int(shape[1]/2):]
    print(plot2.shape)
    print(plot1==plot2)
    plt.plot(plot2)
    plt.show()

    plot3 = df[int(shape[0]/2):, 0:int(shape[1]/2)]
    print(plot3.shape)
    plt.plot(plot3)
    plt.show()

    plot4 = df[int(shape[0]/2):, int(shape[1]/2):]
    print(plot4.shape)
    print(plot3==plot4)
    plt.plot(plot4)
    plt.show()


def nine_plots():
    ni = int(df.shape[0]/9)
    fi = int(df.shape[1]/4)

    fig, ax = plt.subplots(3, 3)
    sub1 = []
    for i in range(0, 9):
        sub1.append(df[i*ni:(i+1)*ni, 0:fi])

    ax[0, 0].plot(sub1[0])
    ax[0, 1].plot(sub1[1])
    ax[0, 2].plot(sub1[2])
    ax[1, 0].plot(sub1[3])
    ax[1, 1].plot(sub1[4])
    ax[1, 2].plot(sub1[5])
    ax[2, 0].plot(sub1[6])
    ax[2, 1].plot(sub1[7])
    ax[2, 2].plot(sub1[8])
    plt.show()

    fig, ax = plt.subplots(3, 3)
    sub2 = []
    for i in range(0, 9):
        sub2.append(df[i*ni:(i+1)*ni, fi:int(fi*2)])

    ax[0, 0].plot(sub2[0])
    ax[0, 1].plot(sub2[1])
    ax[0, 2].plot(sub2[2])
    ax[1, 0].plot(sub2[3])
    ax[1, 1].plot(sub2[4])
    ax[1, 2].plot(sub2[5])
    ax[2, 0].plot(sub2[6])
    ax[2, 1].plot(sub2[7])
    ax[2, 2].plot(sub2[8])
    plt.show()

    fig, ax = plt.subplots(3, 3)
    sub3 = []
    for i in range(0, 9):
        sub3.append(df[i*ni:(i+1)*ni, int(fi*2):int(fi*3)])

    ax[0, 0].plot(sub3[0])
    ax[0, 1].plot(sub3[1])
    ax[0, 2].plot(sub3[2])
    ax[1, 0].plot(sub3[3])
    ax[1, 1].plot(sub3[4])
    ax[1, 2].plot(sub3[5])
    ax[2, 0].plot(sub3[6])
    ax[2, 1].plot(sub3[7])
    ax[2, 2].plot(sub3[8])
    plt.show()

    fig, ax = plt.subplots(3, 3)
    sub4 = []
    for i in range(0, 9):
        sub4.append(df[i*ni:(i+1)*ni, int(fi*3):int(fi*4)])

    ax[0, 0].plot(sub4[0])
    ax[0, 1].plot(sub4[1])
    ax[0, 2].plot(sub4[2])
    ax[1, 0].plot(sub4[3])
    ax[1, 1].plot(sub4[4])
    ax[1, 2].plot(sub4[5])
    ax[2, 0].plot(sub4[6])
    ax[2, 1].plot(sub4[7])
    ax[2, 2].plot(sub4[8])
    plt.show()


def one_after_another(amps):
    data = filter_data(amps)
    data = np.array(data)
    t_data = data.transpose()
    print("data shape:", data.shape)
    mean = np.mean(t_data, axis=0)
    print(mean.shape)

    plt.plot(t_data)
    plt.show()

    # ty = int(data.shape[0] / 20)
    # for i in range(19):
    #     plt.plot(mean[i*ty:(i+1)*ty])
    #     plt.show()


df = parse("kanalforsøg/Kanal11Bevægelse.csv")
# amps = amp_calc(df)
# amps = remove_bad_carriers(amps)
# print(df.shape)
# one_after_another(amps)
#
# df = parse("kanalforsøg/Kanal11Stilhed.csv")
# amps = amp_calc(df)
# amps = remove_bad_carriers(amps)
# print(df.shape)
# one_after_another(amps)

# still1 = amp_calc(parse("kanalforsøg/Kanal1Stilhed.csv"))
# still6 = amp_calc(parse("kanalforsøg/Kanal6Stilhed.csv"))
# still11 = amp_calc(parse("kanalforsøg/Kanal11Stilhed.csv"))
#
# move1 = amp_calc(parse("kanalforsøg/Kanal1Bevægelse.csv"))
# move6 = amp_calc(parse("kanalforsøg/Kanal6Bevægelse.csv"))
# move11 = amp_calc(parse("kanalforsøg/Kanal11Bevægelse.csv"))

# labels = [0]*len(still1) + [1]*len(move1) + [0]*len(still6) + [1]*len(move6) + [0]*len(still11) + [1]*len(move11)
# data = np.concatenate((still1, move1, still6, move6, still11, move11), axis=0)
# print(len(labels))
# print(len(data))

still1 = amp_calc(parse("Bib_kanal_test/bib_kanal9_still.csv"))
still2 = amp_calc(parse("Bib_kanal_test/bib_kanal9_still2.csv"))
move1 = amp_calc(parse("Bib_kanal_test/bib_kanal9_move.csv"))
move2 = amp_calc(parse("Bib_kanal_test/bib_kanal9_move2.csv"))

still = np.concatenate((remove_bad_carriers(still1), remove_bad_carriers(still2)), axis=0)
move = np.concatenate((remove_bad_carriers(move1), remove_bad_carriers(move2)), axis=0)

still_filtered = filter_data(still)
move_filtered = filter_data(move)

plt.subplot(2, 1, 1)
plt.axvline(x=len(still1), linestyle='dotted')
plt.plot(still_filtered)
plt.title("Bib still concatenated")
plt.subplot(2, 1, 2)
plt.axvline(x=len(move1), linestyle='dotted')
plt.plot(move_filtered)
plt.title("Bib move concatenated")
plt.show()

