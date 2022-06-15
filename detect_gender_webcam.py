from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv


class GenderDetection:
    def __init__(self):
        # load model
        self.model = load_model('exported_model/')
        self.classes = ['man','woman']
        self.directory = os.getcwd()
    
    def imageFile(self, image):
        img_path = self.directory + "\images\\"+ image
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        face, confidence = cv.detect_face(img)

        # loop through detected faces
        for idx, f in enumerate(face):
            # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)

            # crop the detected face region
            face_crop = np.copy(img[startY:endY,startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96,96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = self.model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

            # get label with max accuracy
            idx = np.argmax(conf)
            label = self.classes[idx]

            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write label and confidence above face rectangle
            cv2.putText(img, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # display output
            cv2.imshow("gender detection", img)

        cv2.waitKey(0)
        # release resources
        cv2.destroyAllWindows()

    def videoCaptureFromCamera(self):
        # open webcam
        webcam = cv2.VideoCapture(0)
        # loop through frames
        while webcam.isOpened():
            # read frame from webcam 
            status, frame = webcam.read()
            # apply face detection
            face, confidence = cv.detect_face(frame)

            # loop through detected faces
            for idx, f in enumerate(face):

                # get corner points of face rectangle        
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

                # crop the detected face region
                face_crop = np.copy(frame[startY:endY,startX:endX])

                if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue

                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop, (96,96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # apply gender detection on face
                conf = self.model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

                # get label with max accuracy
                idx = np.argmax(conf)
                label = self.classes[idx]

                label = "{}: {:.2f}%".format(label, conf[idx] * 100)

                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write label and confidence above face rectangle
                cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            # display output
            cv2.imshow("gender detection", frame)

            # press "Q" to stop.
            key = cv2.waitKey(1)
            if key == 27:
                break

        # release resources
        webcam.release()
        cv2.destroyAllWindows()




# """  Driver Code Or Main  """
if __name__ == "__main__":
    genderDetection = GenderDetection()


    # Press Esc to exit from loop 
    genderDetection.videoCaptureFromCamera()

    # Give image name from images directory
    #genderDetection.imageFile('img7.jfif')