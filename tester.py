import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import time
import numpy as np

matplotlib.use('TkAgg')

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (3.2, 2.4)  # Number of hundreds of pixels as float
while True:
    plt.plot([1, 2, 3, 4], color='r')
    plt.ylabel('some numbers')
    start = time.process_time()
    img_buff = io.BytesIO()
    plt.savefig(img_buff, format='png')

    im = Image.open(img_buff)
    data = np.asarray(im)
    # im.show(title="Image test")
    cv2.imshow("TEST", data)
    if cv2.waitKey(1) == ord('q'):  # q to quit
        break
    end = time.process_time()
    print("Done in " + str(end-start) + " seconds")

