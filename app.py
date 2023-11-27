from flask import Flask, request, jsonify, send_from_directory
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
os.makedirs(config["output_folder"], exist_ok=True)

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

    files = request.files.getlist('files')
    result = []

    for file in files:
        try:
            image = cv2.imdecode(numpy.frombuffer(
                file.read(), numpy.uint8), cv2.IMREAD_COLOR)

            face_locations = face_recognition.face_locations(image)
            face_encoding = face_recognition.face_encodings(
                image, face_locations)[0]

            mongo.db.faceDescriptors.insert_one({
                "label": label,
                "descriptor": face_encoding.tolist()
            })
            result.append({
                "label": label,
                "path": file.filename
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

    image = cv2.imread(local_file_path)
    result, frame = recognizeInImage(image)

    cv2.imwrite(local_file_path, frame)
    return result


@app.route('/api/recognize/image', methods=['POST'])
def recognize_image():
    file = request.files['file']
    image = cv2.imdecode(numpy.frombuffer(
        file.read(), numpy.uint8), cv2.IMREAD_COLOR)
    result = recognizeInImage(file.filename, image)

    return jsonify({
        "predict": result,
        "output": file.filename
    })


@app.route('/api/download/<file_name>', methods=['GET'])
def download(file_name):
    print(file_name)
    return send_from_directory(config["output_folder"], file_name, as_attachment=True)


def recognizeInImage(file_name, frame):
    result = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(
        frame, face_locations)
    for index, face_encoding in enumerate(face_encodings):
        top, right, bottom, left = face_locations[index]

        face_distances = face_recognition.face_distance(
            known_face_descriptors, face_encoding)
        min_distance = min(face_distances)
        if min_distance < 0.3:
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

    cv2.imwrite(os.path.join(config["output_folder"], file_name), frame)
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=config["port"])
