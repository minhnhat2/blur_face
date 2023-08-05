import cv2
import os
import numpy as np


import face_recognition

def get_most_confident_match(face_encoding, known_encodings, known_names):
    # Compare the input face encoding with the list of known encodings
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

    # Find the index of the face with the smallest distance (most confident match)
    best_match_index = int(np.argmin(face_distances))

    # Return the name of the most confident match
    if matches[best_match_index]:
        return known_names[best_match_index]
    else:
        return None

def blur_faces_in_camera(known_encodings, known_names):
    # Open the default camera (0) or any external camera (1, 2, etc.)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Resize frame to speed up face detection processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up face locations to the original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Get the name of the most confident match
            name = get_most_confident_match(face_encoding, known_encodings, known_names)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw the name below the face
            cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom + 20), font, 0.8, (255, 255, 255), 1)

            # If the face belongs to a known individual, blur the face
            if name in known_names:
                face_image = frame[top:bottom, left:right]
                face_image = cv2.GaussianBlur(face_image, (75, 75), 0)
                frame[top:bottom, left:right] = face_image

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Load the face encodings and names from the images directory
    known_encodings = []  # List of face encodings for known individuals
    known_names = []  # List of names corresponding to the known encodings
    images_directory = "images"  # Directory containing the images with names

    for root, dirs, files in os.walk(images_directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(root, file)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]  # Assuming only one face per image
                name = os.path.splitext(file)[0]  # Extract name from the filename
                known_encodings.append(encoding)
                known_names.append(name)

    print('Press "q" to quit.')
    blur_faces_in_camera(known_encodings, known_names)
