import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def CarrierPlot(phase,amp,start,stop):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    x = [i for i in range(len(amp))]

    plt.title("Line graph")

    for i in range(start, stop):
        plt.plot(x, amp[:, i])
    plt.xlabel("Packet number/Tid")
    plt.ylabel("Amplitude")
    plt.show()

