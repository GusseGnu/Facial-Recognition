from imutils import paths
import face_recognition
import pickle
import cv2
import os
import time
import numpy
import math
import multiprocessing
import functools

file_exists = os.path.exists("face_enc")
knownEncodings = []
knownNames = []


def training():

    if not file_exists:

        t1_start = time.process_time()

        # get paths of each file in folder named Training Images
        # Training Images here contains my data(folders of various persons)
        imagePaths = list(paths.list_images('Training Images'))
        print("imagePaths: " + str(imagePaths))
        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            print("imagePath: \"" + str(imagePath) + "\"")
            # extract the person name from the image path
            name = imagePath.split(os.path.sep)[-2]
            print("name: " + str(name))
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Use Face_recognition to locate faces
            boxes = face_recognition.face_locations(image, model='cnn')
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)
        # save encodings along with their names in dictionary data
        data = {"encodings": knownEncodings, "names": knownNames}
        # use pickle to save data into a file for later use
        f = open("face_enc", "wb")
        f.write(pickle.dumps(data))
        f.close()

        t1_stop = time.process_time()
        print("Completed facial feature extraction in " + str(t1_stop - t1_start) + " seconds with an average of " + str((t1_stop - t1_start) / len(imagePaths)) + " seconds per image")

# ------------------------------------------------------------------------------------------------ #


def image_recognition():

    t2_start = time.process_time()

    # find path of xml file containing haarcascade file
    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('face_enc', "rb").read())
    # Find path to the image you want to detect face and pass it here
    image = cv2.imread("Test Images/Obama_and_I.jpg")
    if image.shape[0] > 720 or image.shape[1] > 720:
        if image.shape[0] >= image.shape[1]:
            image = cv2.resize(image, (0, 0), fx=(720/image.shape[0]), fy=(720/image.shape[0]))
        else:
            image = cv2.resize(image, (0, 0), fx=(720/image.shape[1]), fy=(720/image.shape[1]))
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
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                # rescale the face coordinates
                # draw the predicted face name on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
        t2_stop = time.process_time()
        print("Completed recognition in " + str(t2_stop - t2_start) + " seconds")
        cv2.imshow("Frame", image)
        cv2.waitKey(0)

# ------------------------------------------------------------------------------------------------ #


def video_recognition_slow():

    # find path of xml file containing haarcascade file
    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    faceCascade = cv2.CascadeClassifier(cascPathface)
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('face_enc', "rb").read())

    print("Streaming started")
    video_capture = cv2.VideoCapture(0)
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

        # convert the input frame from BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # the facial embeddings for face in input
        encodings = face_recognition.face_encodings(rgb)
        names = []
        # loop over the facial embeddings incase
        # we have multiple embeddings for multiple fcaes
        for encoding in encodings:
            # Compare encodings with encodings in data["encodings"]
            # Matches contain array with boolean values and True for the embeddings it matches closely
            # and False for rest
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            # set name to Unknown if no encoding matches
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
            # loop over the recognized faces
            for ((x, y, w, h), name) in zip(faces, names):
                # rescale the face coordinates
                # draw the predicted face name on the image
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
        cv2.imshow("Facial recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def video_recognition_fast():
    process_current_frame = True
    video_capture = cv2.VideoCapture(0)  # Camera by index, 0 is default, 0 is laptop camera, 1 is USB
    capture_props = []
    for i in range(10, 15):
        capture_props.append(video_capture.get(i))
    for i in range(0, len(capture_props)):
        print(capture_props[i])
    knownData = pickle.loads(open('face_enc', "rb").read())
    knownEncodings = knownData["encodings"]
    knownNames = knownData["names"]

    if not video_capture.isOpened():
        os.exit('Video source not found...')

    while True:
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_current_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(knownEncodings, face_encoding)
                name = "Unknown"
                confidence = '???'

                # Calculate the shortest distance to face
                face_distances = face_recognition.face_distance(knownEncodings, face_encoding)

                best_match_index = numpy.argmin(face_distances)
                if matches[best_match_index]:
                    name = knownNames[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])

                face_names.append(f'{name} ({confidence})')

        process_current_frame = not process_current_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Create the frame with the name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display the resulting image
        if capture_props == [128.0, 32.0, 32.0, -1.0, 64.0]:
            cv2.imshow('Face Recognition USB camera', frame)
        elif capture_props == [0.0, 50.0, 64.0, 0.0, -1.0]:
            cv2.imshow('Face Recognition laptop camera', frame)
        else:
            cv2.imshow('Face Recognition unknown camera', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


input_queue = multiprocessing.Queue()
output_queue = multiprocessing.Queue()
k = 0

cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
file_exists = os.path.exists("face_enc")
if file_exists:
    data = pickle.loads(open('face_enc', "rb").read())
    knownEncodings = data["encodings"]
    knownNames = data["names"]
else:
    training()
num_processes = multiprocessing.cpu_count()


def video_capture():
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        input_queue.put(frame)
        print("Capturing")
        print(input_queue.qsize())


def video_processing():
    while True:
        print("Pre-processing")
        if not input_queue.qsize() == 0:
            print("Processing")
            frame = input_queue.get()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(knownEncodings, face_encoding)
                name = "Unknown"
                confidence = '???'

                face_distances = face_recognition.face_distance(knownEncodings, face_encoding)
                best_match_index = numpy.argmin(face_distances)
                if matches[best_match_index]:
                    name = knownNames[best_match_index]
                    confidence = face_confidence(face_distances[best_match_index])

                face_names.append(f'{name} ({confidence}')

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                output_queue.put(frame)


def video_display():
    while True:
        cv2.imshow('Face Recognition', output_queue.get())
        print("Displaying")

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


def smap(f):
    return f()


# if __name__ == "__main__":

    # processes = []
    # p1 = multiprocessing.Process(target=video_capture())
    # processes.append(p1)
    # p1.start()
    #
    # p2 = multiprocessing.Process(target=video_processing())
    # processes.append(p2)
    # p2.start()
    #
    # p3 = multiprocessing.Process(target=video_display())
    # processes.append(p3)
    # p3.start()
    #
    # for p in processes:
    #     p.join()

    # p = multiprocessing.Pool(num_processes)
    # res = p.map(smap, [video_capture, video_processing, video_display])
    # p.map_async(video_capture())
    # p.map_async(video_processing())
    # p.map_async(video_display())

    # jobs = []
    # process = multiprocessing.Process(target=video_recognition_fast())
    # jobs.append(process)
    #
    # for j in jobs:
    #     j.start()
    #
    # for j in jobs:
    #     j.join()


training()
# image_recognition()
video_recognition_fast()
