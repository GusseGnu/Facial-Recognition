import time
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib
import pandas as pd
import tensorflow
from tensorflow import keras
matplotlib.use('TkAgg')


data = pd.read_csv('stinky.csv')
data = data.iloc[:, 25]
data = data.to_numpy()
data = data[22:]
print(data.shape)
csi_data = []
all_data = []
good_carriers = []
errors = []


def data_by_package():
    for i in range(0, len(data)):
        data[i] = data[i][1:len(data[i])-2]
        s = data[i].split(' ')
        numbers = [int(j) for j in s]
        # print(numbers)
        csi_data.append(numbers)


def data_by_carrier(carrier):
    temp_data = []
    return_data = []
    for i in range(0, len(data)-1):
        # print("i: " + str(i))
        if data[i][0] == "[":
            data[i] = data[i][1:len(data[i])-1]
        if data[i][len(data[i])-1] == " ":
            data[i] = data[i][0:len(data[i]) - 2]
        elif data[i][len(data[i])-1] == "-":
            data[i] = data[i][0:len(data[i]) - 3]
        s = data[i].split(' ')
        try:
            numbers = [int(j) for j in s]
            temp_data.append(numbers)
            for j in range(0, len(temp_data)-1):
                return_data.append(temp_data[j][carrier])
                all_data.append(temp_data[j][carrier])
        except (ValueError, IndexError) as error:
            errors.append(error)
            print(str(error) + " at index: " + str(i))
    # print(return_data)
    return return_data


def check_if_bad_carrier(process_data):
    if not process_data:
        return True
    counter = 0
    for i in range(1, len(process_data)-1):
        if process_data[i] == process_data[i-1]:
            counter += 1
        else:
            counter = 0
        if counter > 10:
            print("True")
            return True
    print("False")
    return False


def old():
    for j in range(0, 128):
        carrier_data = data_by_carrier(j)
        # print(len(carrier_data))
        # print("j: " + str(j))
        # print(carrier_data)
        if not check_if_bad_carrier(carrier_data):
            csi_data.append(carrier_data)
            good_carriers.append(j)

    print("|----------------------------|")
    print("csi_data len: " + str(len(csi_data)) + " ")
    print("|----------------------------|")
    print("all_data len: " + str(len(all_data)) + " ")
    print("|----------------------------|")
    print("Number of good carriers: " + str(len(good_carriers)))
    print("|----------------------------|")
    print("List of good carriers: " + str(good_carriers))
    print("|----------------------------|")
    print("Number of bad carriers: " + str(128-len(good_carriers)))
    print("|----------------------------|")
    print("Amount of errors: " + str(len(errors)))
    print("|----------------------------|")

    # data_by_package()
    # print(len(csi_data))
    # print(csi_data[100][len(csi_data[100])-1])

    # print(len(csi_data))
    # print(csi_data)

    # data_by_package()
    # print(csi_data)


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


def make_labels(data):
    labels = []
    for i in range(len(data)-1):
        if (data[i+1] > (data[i]*1.4)) or (data[i+1] < (data[i]*0.6)):
            labels.append(1)
        else:
            labels.append(0)
    labels.append(0)
    return labels


new_data = parse("nostink.csv")

# plt.plot(new_data[:300, 320])
# plt.show()

amps = amp_calc(new_data)
print("amps data shape ", amps.shape)
# plt.plot(amps[:300, 1])
# plt.show()

new_amps = np.array(remove_bad_carriers(amps))
print(new_amps.shape)

one_carrier = new_amps[:, 64]
single_feature_normalizer = keras.layers.Normalization(axis=None)
single_feature_normalizer.adapt(one_carrier)
one_carrier = one_carrier.tolist()
# print(len(one_carrier))
labels = make_labels(one_carrier)

# new_amps = new_amps.flatten()
# new_amps = new_amps.tolist()
# print(len(new_amps))

# print(labels)
# print(len(labels))

# plt.plot(new_amps[:300, new_amps.shape[1]-1])
# plt.show()

# labels = make_labels(new_amps)


def train_on_all():


    # model = keras.Sequential([
    #     keras.layers.Dense(167, input_shape=(167,), activation='relu'),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(167, activation='relu'),
    #     keras.layers.Dense(1, activation='softmax')])
    #
    # model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    # model.build()
    # model.summary()
    #
    # new_amps = np.array(new_amps)
    # test = [1]*len(new_amps)
    # test = np.array(test)
    # model.fit(new_amps, test, epochs=10)
    #
    # test_loss, test_acc = model.evaluate(new_amps, test)
    # print('\nTest accuracy:', test_acc*100, "%")


    # train_data = tensorflow.data.Dataset.from_tensor_slices((new_amps, [1]*int(len(new_amps)/2) + [1] + [0]*int(len(new_amps)/2)))
    # train_data = train_data.shuffle(100).batch(64)

    # train_data = np.array(new_amps)
    # print(train_data.shape)
    # train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], 1))
    # print(train_data.shape)
    # train_labels = [1]*int(len(new_amps)/2) + [1] + [0]*int(len(new_amps)/2)
    # train_labels = np.array(train_labels)

    still1 = amp_calc(parse("kanalforsøg/Kanal1Stilhed.csv"))
    still6 = amp_calc(parse("kanalforsøg/Kanal6Stilhed.csv"))
    still11 = amp_calc(parse("kanalforsøg/Kanal11Stilhed.csv"))

    move1 = amp_calc(parse("kanalforsøg/Kanal1Bevægelse.csv"))
    move6 = amp_calc(parse("kanalforsøg/Kanal6Bevægelse.csv"))
    move11 = amp_calc(parse("kanalforsøg/Kanal11Bevægelse.csv"))

    train_labels = [0] * len(still1) + [1] * len(move1) + [0] * len(still6) + [1] * len(move6) + [0] * len(still11) + [
        1] * len(move11)
    train_data = np.concatenate((still1, move1, still6, move6, still11, move11), axis=0)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    print(train_data.shape)

    model = keras.Sequential()
    # model.add(keras.layers.LSTM(167, input_shape=(train_data.shape[1], 1)))
    model.add(keras.layers.Input(shape=(train_data.shape[1], 1), dtype=tensorflow.float64)),
    model.add(keras.layers.LSTM(train_data.shape[1])),
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # model.add(keras.layers.RepeatVector(train_data.shape[0]))
    # model.add(keras.layers.LSTM(train_data.shape[1], return_sequences=True))
    # model.add(keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid')))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build()
    model.summary()

    print(train_data[0][0].dtype)

    model.fit(train_data, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(train_data, train_labels)
    print('\nTest accuracy:', test_acc*100, "%")
    print("Ratio of ones:", train_labels.tolist().count(1) / len(train_labels.tolist()) * 100, "%")
    # probability_model = keras.Sequential([model, keras.layers.Softmax()])
    # predictions = probability_model.predict(test_images)

    test_data = amp_calc(parse("kanalforsøg/Kanal6Bevægelse.csv"))
    predictions = (model.predict(test_data) > 0.5).astype(int)
    print("Nonzero count", np.count_nonzero(predictions))
    print("Test accuracy:", np.count_nonzero(predictions)/len(predictions))

    test_data = amp_calc(parse("kanalforsøg/Kanal1Stilhed.csv"))
    predictions = (model.predict(test_data) > 0.5).astype(int)
    print("Nonzero count", np.count_nonzero(predictions))
    print("Test accuracy:", 1-(np.count_nonzero(predictions)/len(predictions)))

def train_on_one():
    model = keras.Sequential([
        # keras.layers.Flatten(input_shape=(1017, 167)),
        keras.layers.Input(shape=(1,)),
        single_feature_normalizer,
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2)])

    model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.build()
    model.summary()
    model.fit(one_carrier, labels, epochs=5)

    test_loss, test_acc = model.evaluate(one_carrier, labels, verbose=2)
    print('\nTest accuracy:', test_acc * 100, "%")
    print("Ratio of zeros: " + str((labels.count(0)*100)/len(labels)) + " %")

    # probability_model = keras.Sequential([model, keras.layers.Softmax()])
    # predictions = probability_model.predict(test_images)


train_on_all()
# print("Ratio of zeros: " + str(labels.count(0)/len(labels)*100) + " %")
