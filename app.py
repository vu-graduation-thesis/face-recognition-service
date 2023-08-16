from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/detect_face', methods=['POST'])
def detect_face():
    file = request.files['image']
    
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print("face", faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    num_faces = len(faces)
    print('Number of faces detected:', num_faces)

    cv2.imwrite('detected_faces.jpg', image)

    return jsonify({'result': 'success', 'num_faces': num_faces})

if __name__ == '__main__':
    app.run()