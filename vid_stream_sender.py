# run this program on each RPi to send a labelled image stream
import socket
import time
from imutils.video import VideoStream
import imagezmq

sender = imagezmq.ImageSender(connect_to='tcp://82.211.207.249:5555')
rpi_name = socket.gethostname()  # send RPi hostname with each image
picam = VideoStream(0).start()  # Camera by index starting at 0
time.sleep(1.0)  # allow camera sensor to warm up
while True:  # send images as stream until Ctrl-C
    image = picam.read()
    sender.send_image(msg=rpi_name, image=image)

