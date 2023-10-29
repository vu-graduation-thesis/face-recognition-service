import cv2
import numpy as np
import os
from config import config
import aws
import threading
import globalVariable


def start_training_model(student_id):
    print(
        f"Start retraining the model data when there are new student - student_id: {student_id}")
    aws.getResourceFromS3(student_id)

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

    print(
        f"Completed retraining the model data when there are new student - student_id: {student_id}")

    lock = threading.Lock()
    lock.acquire()
    globalVariable.recognizer = recognizer
    lock.release()
