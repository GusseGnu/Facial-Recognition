from imutils import paths
import face_recognition
import pickle
import cv2
import os
import time
import numpy
import math
import multiprocessing
import imagezmq
import socket
from PIL import Image
from PIL import GifImagePlugin


def training():
    if not file_exists:

        t1_start = time.process_time()

        # get paths of each file in training folder
        image_paths = list(paths.list_files('Yalefaces'))
        print("imagePaths: " + str(image_paths))
        # loop over the image paths
        for (i, imagePath) in enumerate(image_paths):
            if not imagePath.__contains__("normal"):
                print("imagePath: \"" + str(imagePath) + "\"")
                # extract the person name from the image path
                name = imagePath.split(os.path.sep)[-2]
                print("name: " + str(name))
                # load the input image and convert it from BGR (OpenCV ordering)
                # to dlib ordering (RGB)
                #image = cv2.imread(imagePath)
                images = Image.open(imagePath)
                images.save("image.jpg")
                image = cv2.imread("image.jpg")
                # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Use Face_recognition to locate faces
                boxes = face_recognition.face_locations(image, model='hog')
                # compute the facial embedding for the face
                encodings = face_recognition.face_encodings(image, boxes)
                # loop over the encodings
                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(name)
        # save encodings along with their names in dictionary data
        face_data = {"encodings": knownEncodings, "names": knownNames}
        # use pickle to save data into a file for later use
        f = open("yale_faces_enc", "wb")
        f.write(pickle.dumps(face_data))
        f.close()

        t1_stop = time.process_time()
        print(
            "Completed facial feature extraction in " + str(t1_stop - t1_start) + " seconds with an average of " + str(
                (t1_stop - t1_start) / len(image_paths)) + " seconds per image")


def testing():
    # get paths of each file in testing folder
    correct_matches = 0
    missed_matches = 0
    image_matches = 0
    image_paths = list(paths.list_images('Yalefaces'))
    print("imagePaths: " + str(image_paths))
    # find path of xml file containing haarcascade file
    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('yale_faces_enc', "rb").read())
    # loop over the image paths
    for (i, imagePath) in enumerate(image_paths):
        if imagePath.__contains__("normal"):
            image_matches += 1
            correct_name = imagePath.split(os.path.sep)[-2]
            # Find path to the image you want to detect face and pass it here
            images = Image.open(imagePath)
            images.save("image.jpg")
            image = cv2.imread("image.jpg")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # convert image to Greyscale for haarcascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,
                                                 scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 minSize=(60, 60),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)

            # the facial embeddings for face in input
            encodings = face_recognition.face_encodings(rgb)
            names = []
            # loop over the facial embeddings incase
            # we have multiple embeddings for multiple fcaes
            for encoding in encodings:
                # Compare encodings with encodings in data["encodings"]
                # Matches contain array with boolean values and True for the embeddings it matches closely
                # and False for rest
                matches = face_recognition.compare_faces(data["encodings"],
                                                         encoding)
                # set name to "Unknown" if no encoding matches
                name = "Unknown"
                # check to see if we have found a match
                if True in matches:
                    # Find positions at which we get True and store them
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # loop over the matched indexes and maintain a count for
                    # each recognized face
                    for i in matchedIdxs:
                        # Check the names at respective indexes we stored in matchedIdxs
                        name = data["names"][i]
                        # increase count for the name we got
                        counts[name] = counts.get(name, 0) + 1
                        # set name which has highest count
                        name = max(counts, key=counts.get)

                    # update the list of names
                    names.append(name)
                if names.__contains__(correct_name):
                    correct_matches += 1
                    print("Matched", name, "correctly")
                else:
                    missed_matches += 1
                    print("Could not match", correct_name)
    print("Missed matches", missed_matches)
    print("Number of matches", correct_matches)
    print("Accuracy", round((correct_matches / image_matches)*100, 2), "%")


# ------------------------------------------------------------------------------------------------ #


def face_confidence(face_distance, face_match_threshold=0.6):  # Helper function used to calculate percentage confidence
    face_range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (face_range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

# ------------------------------------------------------------------------------------------------ #
# Make the facial recognition modular allowing multiprocessing for increased smoothness


def video_capture(in_queue, out_queue, box_queue):
    vid = cv2.VideoCapture(0)
    print("Capturing in " + str(int(vid.get(3))) + "x" + str(int(vid.get(4))) + " at " + str(int(vid.get(5))) + "fps")
    # Keep track of when a face was last recognized
    time_since_last_new = time.process_time()
    while True:
        ret, frame = vid.read()
        # print("in_queue size " + str(in_queue.qsize()))
        # print("out_queue size " + str(out_queue.qsize()))
        # Put frame in queue for processing
        if in_queue.empty():
            in_queue.put_nowait(frame)
        else:
            in_queue.get()
            in_queue.put_nowait(frame)
        if not box_queue.empty():
            # If a face box is ready, and it was found recently, draw ít
            if (box_queue.qsize() == 1) and ((time.process_time() - time_since_last_new) < 0.5):
                # print("Framing")
                box = box_queue.get()
                print("Getting - " + str(box[0]) + ", " + str(box[3]) + ", " + str(box[2]) + ", " +
                      str(box[1]) + " box_queue size: " + str(box_queue.qsize()))
                cv2.rectangle(frame, (box[3], box[0]), (box[1], box[2]), (0, 0, 255), 2)
                cv2.rectangle(frame, (box[3], box[2] - 35), (box[1], box[2]), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, box[4], (box[3] + 6, box[2] - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                box_queue.put(box)
            else:  # If there is a newer facebox, dump the old and restart timer
                box_queue.get()
                time_since_last_new = time.process_time()
        # Put in the queue to be displayed
        out_queue.put_nowait(frame)
        # If out_queue size is growing in size, then the display has closed and so should capture
        if out_queue.qsize() > 10:
            break


def video_processing(in_queue, box_queue):
    while True:
        if not in_queue.empty():
            # print("Processing")
            # Get frame to process from queue, resize it for faster processing, find faces and encode
            frame = in_queue.get_nowait()
            # TODO Instead of resizing entire image, process area around last found face
            # if box_queue.empty():
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            # Load encodings and initialize default name and confidence
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(knownEncodings, face_encoding)
                name = "Unknown"
                confidence = '???'

                # Compare encodings and match to the closest one
                face_distances = face_recognition.face_distance(knownEncodings, face_encoding)
                best_match_index = numpy.argmin(face_distances)
                if matches[best_match_index]:
                    name = knownNames[best_match_index]
                    # Use helper function to calculate a confidence for face to display
                    confidence = face_confidence(face_distances[best_match_index])

                face_names.append(f'{name} ({confidence})')

            # Rescale the position of the found faces (was scaled down to 0.25 so multiply by 4)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                box_queue.put([top, right, bottom, left, name])
                print("top: " + str(top) + " left: " + str(left) + " bottom: " + str(bottom) + " right: " + str(
                    right) + " box_queue size: " + str(box_queue.qsize()))
                # print("box_queue size " + str(box_queue.qsize()))
            # else:
            #     borders = box_queue.get()
            #     box_queue.put(borders)
            #     small_frame = frame[(borders[0]-20):(borders[2]+20), (borders[3]-20):(borders[1]+20)]
            #     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            #     face_locations = face_recognition.face_locations(rgb_small_frame)
            #     face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            #     face_names = []
            #
            #     # Load encodings and initialize default name and confidence
            #     for face_encoding in face_encodings:
            #         matches = face_recognition.compare_faces(knownEncodings, face_encoding)
            #         name = "Unknown"
            #         confidence = '???'
            #
            #         # Compare encodings and match to the closest one
            #         face_distances = face_recognition.face_distance(knownEncodings, face_encoding)
            #         best_match_index = numpy.argmin(face_distances)
            #         if matches[best_match_index]:
            #             name = knownNames[best_match_index]
            #             # Use helper function to calculate a confidence for face to display
            #             confidence = face_confidence(face_distances[best_match_index])
            #
            #         face_names.append(f'{name} ({confidence})')
            #
            #     # Since the frame is not scaled, no action is necessary for rescaling
            #     for (top, right, bottom, left), name in zip(face_locations, face_names):
            #         top += borders[0]-20
            #         right += borders[1]+20
            #         bottom += borders[2]+20
            #         left += borders[3]-20
            #         box_queue.put([top, right, bottom, left, name])
            #         # print("box_queue size " + str(box_queue.qsize()))
            #         print("Putting - " + str(top) + ", " + str(left) + ", " + str(bottom) + ", " +
            #               str(right) + " box_queue size: " + str(box_queue.qsize()))


def video_display(out_queue):
    sender = imagezmq.ImageSender(connect_to='tcp://82.211.207.249:5555')
    rpi_name = socket.gethostname()
    while True:
        # Display video
        image = out_queue.get()
        cv2.imshow('Face Recognition', image)
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        # sender.send_image(msg=rpi_name, image=image)
        # print("Displaying")

        # Close if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


cas_face_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cas_face_path)
file_exists = os.path.exists("yale_faces_enc")
if file_exists:
    data = pickle.loads(open('yale_faces_enc', "rb").read())
    knownEncodings = data["encodings"]
    knownNames = data["names"]
    testing()
else:
    knownEncodings = []
    knownNames = []
    training()
    testing()


# if __name__ == "__main__":

    # input_queue = multiprocessing.Queue(maxsize=1000)
    # output_queue = multiprocessing.Queue(maxsize=1000)
    # facebox_queue = multiprocessing.Queue(maxsize=1000)
    #
    # proc1 = multiprocessing.Process(target=video_capture, args=(input_queue, output_queue, facebox_queue))
    # proc1.start()
    #
    # proc2 = multiprocessing.Process(target=video_processing, args=(input_queue, facebox_queue))
    # proc2.start()
    #
    # proc3 = multiprocessing.Process(target=video_display, args=(output_queue,))
    # proc3.start()
    #
    # # TODO Close all processes when q pressed
