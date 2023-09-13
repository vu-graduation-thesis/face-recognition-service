import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
