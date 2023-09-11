import cv2
import numpy as np
import os
from config import config
import aws
import threading
import globalVariable


def raw_data_processing(trained_student_id=None):
    if trained_student_id is None:
        print(
            f"\nStart preprocessing the raw data for student_id: {trained_student_id}")
    else:
        print(
            f"\nStart preprocessing the raw data for all students")

    raw_folder_data = config['raw_folder_data']
    training_folder_data = config['training_folder_data']

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for student_id in os.listdir(raw_folder_data):
        if trained_student_id is not None and trained_student_id != student_id:
            continue
        student_directory = f"{raw_folder_data}/{student_id}/"

        for image_file in os.listdir(student_directory):
            image_path = student_directory + image_file
            img = cv2.imread(image_path)

            # Convert the image to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces_rect = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5)

            os.makedirs(f'{training_folder_data}/{student_id}', exist_ok=True)

            for (x, y, w, h) in faces_rect:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (100, 100))
                cv2.imwrite(
                    f'{training_folder_data}/{student_id}/{image_file}', face_resized)

    if trained_student_id is None:
        print(
            f"Completed preprocessing the raw data for student_id: {trained_student_id}")
    else:
        print(
            f"Completed preprocessing the raw data for all students")


def start_training_model(student_id=None):
    print(f"\nStart training model for student_id {student_id}")
    aws.getResourceFromS3(student_id="" if student_id is None else student_id)
    raw_data_processing(student_id)

    training_folder_data = config['training_folder_data']
    faces = []
    labels = []

    for student_id in os.listdir(training_folder_data):
        student_directory = f"{training_folder_data}/{student_id}/"

        for image_file in os.listdir(student_directory):
            img = cv2.imread(student_directory + image_file)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces.append(gray_image)
            labels.append(int(student_id))

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(config["trained_model"])
    print(f"Training model for student_id {student_id} is done.")

    lock = threading.Lock()
    lock.acquire()
    globalVariable.recognizer = recognizer
    lock.release()
