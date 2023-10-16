from flask import Flask, request, jsonify
import cv2
import numpy as np
import listener_queue
from threading import Thread, Lock
import globalVariable
import os
from config import config
import training
from PIL import Image
from io import BytesIO
import base64

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



if not os.path.isfile(config["trained_model"]):
    print(f"File {config['trained_model']} is missing. Start training model.")
    training.start_training_model()

print(f"Loading trained model from {config['trained_model']}...")
globalVariable.recognizer.read(config["trained_model"])

os.makedirs("uploads", exist_ok=True)


@app.route('/')
def index():
    return 'Hello world'


@app.route('/upload', methods=['POST'])
def upload():
    if 'type' in request.form and request.form['type'] == 'base64':
        image_data = request.form['image']
        image_data = image_data.split(",")[1]
        img = Image.open(BytesIO(base64.b64decode(image_data)))
        img = img.convert('RGB')
        img.save(os.path.join('uploads', 'image.jpg'))
        return 'File base64 saved successfully', 200
    print(request.files)
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']

    if file:
        filename = file.filename
        file.save(os.path.join('uploads', filename))
        return 'File saved successfully', 200


@app.route('/detect-face', methods=['POST'])
@cross_origin()
def detect_face():

    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(
        file.read(), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite("uploads/image.jpg", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = globalVariable.face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    lock = Lock()
    lock.acquire()
    local_recognizer = globalVariable.recognizer
    lock.release()

    recognized_students = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))

        label, confidence = local_recognizer.predict(face_resized)
        print("label", label, confidence)
        if confidence < 120:
            recognized_students.append({
                "label": label,
                "confidence": confidence,
                "position": {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                }
            })

    return jsonify({'result': 'success', 'num_faces': recognized_students})


if __name__ == '__main__':
    redis_thread = Thread(target=listener_queue.message_listener)
    redis_thread.start()
    app.run(host='0.0.0.0', debug=False, port=config["port"])
    redis_thread.join()
