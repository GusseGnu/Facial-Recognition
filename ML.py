import time
import matplotlib.pyplot as plt
import math
import pywt
import numpy as np
import matplotlib
import pandas as pd
import tensorflow
from tensorflow import keras
from sklearn import feature_selection
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


def amp_calc(csi_data):
    amplitude = np.array([np.sqrt(data[::2]**2 + data[1::2]**2) for data in csi_data])
    return amplitude


def remove_bad_carriers(amp_data):
    good_ones = amp_data
    indices = []
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
                indices.append(i)
                n += 1
                print("Deleted " + str(i+1))
                break
    return good_ones, indices


def make_labels(data):
    labels = []
    for i in range(len(data)-1):
        if (data[i+1] > (data[i]*1.4)) or (data[i+1] < (data[i]*0.6)):
            labels.append(1)
        else:
            labels.append(0)
    labels.append(0)
    return labels


def train_model(train_data, train_labels, epochs, config):
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    print(train_data.shape)
    print("Data type is", train_data[0][0].dtype)

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(train_data.shape[1], 1), dtype=tensorflow.float64)),
    if "regularizer" in config.lower():
        model.add(keras.layers.LSTM(train_data.shape[1], kernel_regularizer=keras.regularizers.L2(0.001))),
    else:
        model.add(keras.layers.LSTM(train_data.shape[1])),
    if "dropout" in config.lower():
        model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build()
    model.summary()

    model.fit(train_data, train_labels, epochs=epochs)
    test_loss, test_acc = model.evaluate(train_data, train_labels)
    print('\nTrain accuracy:', test_acc*100, "%")
    return model, test_acc*100


def repeat_model_train(train_data, train_labels, test_data, test_labels, repeats, epochs, config):
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    print("Train data shape:", train_data.shape, "Test data shape:", test_data.shape)
    train_accuracies = []
    test_accuracies = []
    for i in range(repeats):
        model, test_acc = train_model(train_data, train_labels, epochs, config)
        train_accuracies.append(test_acc)
        predictions = (model.predict(test_data) > 0.5).astype(int)
        count = 0
        for j in range(len(predictions)):
            if predictions[j] == test_labels[j]:
                count += 1
        accuracy = (count / len(test_labels)) * 100
        print("Test accuracy:", accuracy, "%")
        test_accuracies.append(accuracy)
    print("Average train accuracy:", round(sum(train_accuracies) / len(train_accuracies), 2), "%")
    print(test_accuracies)
    print("Average test accuracy:", round(sum(test_accuracies)/len(test_accuracies), 2), "%")


def feature_select(train_data, train_labels, amount, show):
    selector = feature_selection.SelectKBest(feature_selection.f_classif, k=10)
    selected_features = selector.fit_transform(train_data, train_labels)
    top_features = (-selector.scores_).argsort()[:amount]
    train_data = np.array(train_data)[:, top_features]
    print(top_features)
    if show:
        plt.subplot(2, 1, 1)
        plt.plot(selector.scores_)
        plt.title("Subcarriers scored by feature selection with suggested threshold")
        print(train_data.shape)
        plt.plot(list(range(0, len(selector.scores_))), [200]*len(selector.scores_), ':')
        # plt.grid(axis='x')
        plt.subplot(2, 1, 2)
        plt.plot(train_data)
        plt.axvline(x=len(1000), linestyle='dotted') #Hardcoded value here, should be len(train_still)
        plt.title("Selected best carriers with seperated classes")
        plt.show()
    return train_data, top_features


def threshold_determination(model, test_data, test_labels):
    best = []
    for i in range(10):
        threshold = 0.05
        to_print = []
        accuracies = []
        predictions = model.predict(test_data)
        for i in range(19):
            rounded = (predictions > threshold).astype(int)
            count = 0
            for j in range(len(rounded)):
                if rounded[j] == test_labels[j]:
                    count += 1
            accuracy = (count/len(test_labels))*100
            print("Test accuracy:", accuracy, "%")
            accuracies.append(accuracy)
            to_print.append((str(round(threshold, 2)) + ": " + str(accuracy) + " %"))
            threshold += 0.05
        best.append(str(max(accuracies)) + " index: " + str(accuracies.index(max(accuracies))))

        for i in range(19):
            print(to_print[i])
    print(best)


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


def train_on_one():
    new_data = parse("nostink.csv")

    amps = amp_calc(new_data)
    print("amps data shape ", amps.shape)

    new_amps, indices = np.array(remove_bad_carriers(amps))
    print(new_amps.shape)

    one_carrier = new_amps[:, 64]
    single_feature_normalizer = keras.layers.Normalization(axis=None)
    single_feature_normalizer.adapt(one_carrier)
    one_carrier = one_carrier.tolist()
    labels = make_labels(one_carrier)

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


def data_config1():     # Baseline observations/test
    train_still = amp_calc(parse("kanal1/train_still.csv"))
    train_move = amp_calc(parse("kanal1/train_move.csv"))
    train_data = np.concatenate((train_still, train_move), axis=0)
    train_data, indices = remove_bad_carriers(train_data)
    train_data = np.array(train_data)
    train_labels = np.array([0]*len(train_still) + [1]*len(train_move))

    test_still = amp_calc(parse("kanal1/test_still.csv"))
    test_move = amp_calc(parse("kanal1/test_move.csv"))
    test_data = np.concatenate((test_still, test_move), axis=0)
    # test_data = np.array(test_data)[:, indices])   # if feature_select then this else if remove_features then np.delete
    # test_data = np.delete(test_data, indices, 1)
    test_labels = np.array([0]*len(test_still) + [1]*len(test_move))

    train_data, indices = feature_select(train_data, train_labels, 66, False)
    test_data = np.array(test_data[:, indices])

    return train_data, train_labels, test_data, test_labels


def data_config2():     # Multiple observations from same room and same channel, but different positioning
    still1 = amp_calc(parse("kanal1/still1.csv"))
    still2 = amp_calc(parse("kanal1/still2.csv"))
    still3 = amp_calc(parse("kanal1/still3.csv"))
    still4 = amp_calc(parse("kanal1/still4.csv"))
    still5 = amp_calc(parse("kanal1/still5.csv"))
    still6 = amp_calc(parse("kanal1/still6.csv"))

    move1 = amp_calc(parse("kanal1/move1.csv"))
    move2 = amp_calc(parse("kanal1/move2.csv"))
    move3 = amp_calc(parse("kanal1/move3.csv"))
    move4 = amp_calc(parse("kanal1/move4.csv"))
    move5 = amp_calc(parse("kanal1/move5.csv"))
    move6 = amp_calc(parse("kanal1/move6.csv"))

    train_data = np.concatenate((still1, move1, still2, move2, still3, move3,
                                 still4, move4, still5, move5, still6, move6), axis=0)
    train_labels = [0]*len(still1)+[1]*len(move1)+[0]*len(still2)+[1]*len(move2)+[0]*len(still3)+[1]*len(move3) \
                   + [0]*len(still4)+[1]*len(move4)+[0]*len(still5)+[1]*len(move5)+[0]*len(still6)+[1]*len(move6)
    return train_data, train_labels


def data_config3():     # For testing carrier intervals
    train_still = amp_calc(parse("kanal1/train_still.csv"))
    train_move = amp_calc(parse("kanal1/train_move.csv"))
    train_data = np.concatenate((train_still, train_move), axis=0)
    train_data = np.array(train_data[:, range(128, 192)])
    train_data, indices = remove_bad_carriers(train_data)
    train_data = np.array(train_data)
    train_labels = np.array([0]*len(train_still) + [1]*len(train_move))

    test_still = amp_calc(parse("kanal1/test_still.csv"))
    test_move = amp_calc(parse("kanal1/test_move.csv"))
    test_data = np.concatenate((test_still, test_move), axis=0)
    test_data = np.array(test_data[:, range(128, 192)])
    test_data = np.delete(test_data, indices, 1)
    test_labels = np.array([0]*len(test_still) + [1]*len(test_move))

    # train_data, indices = feature_select(train_data, train_labels, 66, False)
    return train_data, train_labels, test_data, test_labels


def data_config4():     # For testing channels and filtering
    train_still = amp_calc(parse("Bib_kanal_test/bib_kanal9_still2.csv"))
    test_still = train_still[int(len(train_still)*0.7):len(train_still), :]
    train_still = train_still[:int(len(train_still)*0.7), :]
    train_move = amp_calc(parse("Bib_kanal_test/bib_kanal9_move2.csv"))
    test_move = train_move[int(len(train_move)*0.7):len(train_move), :]
    train_move = train_move[:int(len(train_move)*0.7), :]
    train_data = np.concatenate((train_still, train_move), axis=0)
    train_data, indices = remove_bad_carriers(train_data)
    train_data = np.array(train_data)
    train_labels = np.array([0]*len(train_still) + [1]*len(train_move))

    test_data = np.concatenate((test_still, test_move), axis=0)
    test_data = np.delete(test_data, indices, 1)
    test_labels = np.array([0]*len(test_still) + [1]*len(test_move))

    train_data = denoise_data(train_data, 0.7754)
    test_data = denoise_data(test_data, 0.7754)
    test_data = np.delete(test_data, len(test_data)-1, axis=0)

    # train_data, indices = feature_select(train_data, train_labels, 66, False)
    return train_data, train_labels, test_data, test_labels


def data_config5():     # For testing channels with separate train and test sets
    train_still = amp_calc(parse("Bib_kanal_test/bib_kanal9_still.csv"))
    train_move = amp_calc(parse("Bib_kanal_test/bib_kanal9_move.csv"))
    train_data = np.concatenate((train_still, train_move), axis=0)
    train_data, indices = remove_bad_carriers(train_data)
    train_data = np.array(train_data)
    train_labels = np.array([0]*len(train_still) + [1]*len(train_move))

    test_still = amp_calc(parse("Bib_kanal_test/bib_kanal9_still2.csv"))
    test_move = amp_calc(parse("Bib_kanal_test/bib_kanal9_move2.csv"))
    test_data = np.concatenate((test_still, test_move), axis=0)
    # test_data = np.array(test_data)[:, indices])   # if feature_select then this else if remove_features then np.delete
    test_data = np.delete(test_data, indices, 1)
    test_labels = np.array([0]*len(test_still) + [1]*len(test_move))

    # train_data, indices = feature_select(train_data, train_labels, 66, False)
    # test_data = np.array(test_data[:, indices])

    return train_data, train_labels, test_data, test_labels


def data_config6():     # For testing filtering and filter values
    train_still = amp_calc(parse("kanal1/train_still.csv"))
    train_move = amp_calc(parse("kanal1/train_move.csv"))
    train_data = np.concatenate((train_still, train_move), axis=0)
    train_data, indices = remove_bad_carriers(train_data)
    train_data = np.array(train_data)
    train_labels = np.array([0]*len(train_still) + [1]*len(train_move))

    test_still = amp_calc(parse("kanal1/test_still.csv"))
    test_move = amp_calc(parse("kanal1/test_move.csv"))
    test_data = np.concatenate((test_still, test_move), axis=0)
    test_data = np.delete(test_data, indices, 1)
    test_labels = np.array([0]*len(test_still) + [1]*len(test_move))

    # train_data, indices = feature_select(train_data, train_labels, 66, False) # if feature_select then this else np.delete
    # test_data = np.array(test_data[:, indices])

    train_data = denoise_data(train_data, 0.7754)
    test_data = denoise_data(test_data, 0.7754)

    return train_data, train_labels, test_data, test_labels


train_data, train_labels, test_data, test_labels = data_config6()
repeat_model_train(train_data, train_labels, test_data, test_labels, 10, 10, "dropout regularizer")
