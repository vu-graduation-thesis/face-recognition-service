import cv2
import face_recognition
import os
import dlib
import numpy

# Load the known student images
known_students_dir = "download/2000"
known_students_images = []

for student_image in os.listdir(known_students_dir):
    image = face_recognition.load_image_file(
        os.path.join(known_students_dir, student_image))
    face_encoding = face_recognition.face_encodings(
        image)[0]  # Assuming only one face per image
    known_students_images.append(face_encoding)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

# Load the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Resize the frame to speed up face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    # rgb_small_frame = small_frame[:, :, ::-1]
    rgb_small_frame = numpy.ascontiguousarray(small_frame[:, :, ::-1])

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Compare the unknown face with known student images
        matches = face_recognition.compare_faces(
            known_students_images, face_encoding)
        name = "Unknown"

        # If a match is found, select the first one
        if True in matches:
            first_match_index = matches.index(True)
            print(first_match_index)

            name = os.listdir(known_students_dir)[
                first_match_index].split('.')[0]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the recognized face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Write the name below the face rectangle
        cv2.rectangle(frame, (left, bottom + 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()


face_locations = face_recognition.face_locations(image)
     face_encodings = face_recognition.face_encodings(image, face_locations)
      face_names = []
       for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                know_face_descriptors, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = know_face_labels[first_match_index]
            face_names.append(name)
        return jsonify({
            "message": "Recognize success",
            "data": face_names
        })
