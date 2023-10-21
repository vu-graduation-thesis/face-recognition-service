from flask import Flask, request, jsonify
import os
from config import config
from flask_pymongo import PyMongo
import aws
import cv2
import face_recognition
import numpy

from flask_cors import CORS, cross_origin
app = Flask(__name__)
app.config["MONGO_URI"] = config["mongodb"]
cors = CORS(app)
os.makedirs(config["download_folder"], exist_ok=True)

mongo = PyMongo(app)

known_face_descriptors = []
known_face_labels = []


def init():
    global known_face_descriptors
    global known_face_labels

    for data in mongo.db.faceDescriptors.find():
        descriptor = numpy.array(data["descriptor"])
        known_face_descriptors.append(descriptor)
        known_face_labels.append(data["label"])


init()


@app.route('/')
def index():
    return 'Hello world'


@app.route('/api/training/<label>', methods=['POST'])
def training_data(label):
    global known_face_descriptors
    global known_face_labels
    body = request.json

    paths = aws.download_folder_from_s3(
        body["bucket"], body["folder_path"])

    result = []

    for path in paths:
        try:
            image = cv2.imread(path)

            face_locations = face_recognition.face_locations(image)
            face_encoding = face_recognition.face_encodings(
                image, face_locations)[0]

            mongo.db.faceDescriptors.insert_one({
                "label": label,
                "descriptor": face_encoding.tolist()
            })
            result.append({
                "label": label,
                "path": path
            })
            known_face_descriptors.append(face_encoding)
            known_face_labels.append(label)
        except Exception as e:
            print("Error training data: ", e)

    return jsonify({
        "message": "Training data success",
        "data": result
    })


@app.route('/api/recognize', methods=['POST'])
def recognize():
    body = request.json
    global known_face_descriptors
    global known_face_labels
    file_path = body["file"]
    type = body["type"]
    bucket = body["bucket"]
    local_file_path = aws.download_file_from_s3(bucket, file_path)
    if local_file_path is None:
        return jsonify({
            "message": "Error downloading image from S3"
        }), 500

    result = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    if type == "image":
        frame = cv2.imread(local_file_path)
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(
            frame, face_locations)
        for index, face_encoding in enumerate(face_encodings):
            top, right, bottom, left = face_locations[index]

            face_distances = face_recognition.face_distance(
                known_face_descriptors, face_encoding)
            min_distance = min(face_distances)
            if min_distance < 0.5:
                index = face_distances.tolist().index(min_distance)
                label = known_face_labels[index]
                print(label)
                result.append({
                    "label": label,
                    "confidence": 1 - min_distance
                })
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (124, 252, 0), 2)
                cv2.putText(frame, label, (left + 6, bottom - 12),
                            font, min(right - left, 35)/(55), (124, 252, 0), 1)
            else:
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left + 6, bottom - 12),
                            font, min(right - left, 35)/(55), (0, 0, 255), 1)

    cv2.imwrite(local_file_path, frame)
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=config["port"])
