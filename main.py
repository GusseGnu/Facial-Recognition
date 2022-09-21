from imutils import paths
import face_recognition
import pickle
import cv2
import os
import time
import numpy
import math
import multiprocessing


def training():
    if not file_exists:

        t1_start = time.process_time()

        # get paths of each file in folder named Training Images
        image_paths = list(paths.list_images('Training Images'))
        print("imagePaths: " + str(image_paths))
        # loop over the image paths
        for (i, imagePath) in enumerate(image_paths):
            print("imagePath: \"" + str(imagePath) + "\"")
            # extract the person name from the image path
            name = imagePath.split(os.path.sep)[-2]
            print("name: " + str(name))
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Use Face_recognition to locate faces
            boxes = face_recognition.face_locations(image, model='cnn')
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)
        # save encodings along with their names in dictionary data
        face_data = {"encodings": knownEncodings, "names": knownNames}
        # use pickle to save data into a file for later use
        f = open("face_enc", "wb")
        f.write(pickle.dumps(face_data))
        f.close()

        t1_stop = time.process_time()
        print(
            "Completed facial feature extraction in " + str(t1_stop - t1_start) + " seconds with an average of " + str(
                (t1_stop - t1_start) / len(image_paths)) + " seconds per image")


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
    # Keep track of when a face was last recognized
    time_since_last_new = time.process_time()
    while True:
        ret, frame = vid.read()
        # Put frame in queue for processing
        if in_queue.empty():
            in_queue.put_nowait(frame)
        if not box_queue.empty():
            # If a face box is ready, and it was found recently, draw ít
            if (box_queue.qsize() == 1) and ((time.process_time() - time_since_last_new) < 0.5):
                print("Framing")
                print(time.process_time() - time_since_last_new)
                box = box_queue.get()
                cv2.rectangle(frame, (box[3], box[0]), (box[1], box[2]), (0, 0, 255), 2)
                cv2.rectangle(frame, (box[3], box[2] - 35), (box[1], box[2]), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, box[4], (box[3] + 6, box[2] - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                box_queue.put(box)
            else:  # If there is a newer facebox, dump the old and restart timer
                box_queue.get()
                time_since_last_new = time.process_time()
        # Put in the queue to be displayed
        out_queue.put_nowait(frame)


def video_processing(in_queue, box_queue):
    while True:
        if not in_queue.empty():
            print("Processing")
            # Get frame to process from queue, resize it for faster processing, find faces and encode
            frame = in_queue.get_nowait()
            # TODO Instead of resizing entire image process area around last found face
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

                face_names.append(f'{name} ({confidence}')

            # Rescale the position of the found faces (was scaled down to 0.25 so multiply by 4)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                box_queue.put([top, right, bottom, left, name])

                # Draw box around recognized face with a name and confidence
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)


def video_display(out_queue):
    while True:
        # Display video
        cv2.imshow('Face Recognition', out_queue.get())
        print("Displaying")

        # Close if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cas_face_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cas_face_path)
    file_exists = os.path.exists("face_enc")
    if file_exists:
        data = pickle.loads(open('face_enc', "rb").read())
        knownEncodings = data["encodings"]
        knownNames = data["names"]
    else:
        training()

    input_queue = multiprocessing.Queue(maxsize=1000)
    output_queue = multiprocessing.Queue(maxsize=1000)
    facebox_queue = multiprocessing.Queue(maxsize=1000)

    proc1 = multiprocessing.Process(target=video_capture, args=(input_queue, output_queue, facebox_queue))
    proc1.start()

    proc2 = multiprocessing.Process(target=video_processing, args=(input_queue, facebox_queue))
    proc2.start()

    proc3 = multiprocessing.Process(target=video_display, args=(output_queue,))
    proc3.start()

    # TODO Close all processes when q pressed
