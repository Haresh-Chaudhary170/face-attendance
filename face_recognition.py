import os
import cv2
import dlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from tensorflow.keras.models import load_model

# Load the trained model and person names
model = load_model('trained_model2.h5')
names = ['Haresh', 'Kushal', 'Potter', 'Shreya']

# Load the face detector from dlib
hog_face_detector = dlib.get_frontal_face_detector()

# Preprocess the input image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200))
    normalized = resized / 255.0
    preprocessed = normalized.reshape(1, 200, 200, 1)  # Use 1 channel for grayscale images
    return preprocessed

# Perform face recognition on an input image
def recognize_face(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    person_index = np.argmax(predictions)
    recognized_person = names[person_index]
    return recognized_person

# Initialize the video capture
video = cv2.VideoCapture(0)

# Get today's date for the CSV filename
today_date = date.today().strftime('%Y-%m-%d')

# Attendance folder path
attendance_folder = os.path.join(os.getcwd(), 'Attendance')

# Create the Attendance folder if it doesn't exist
if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

# Load attendance data from existing CSV file if available
csv_filename = os.path.join(attendance_folder, f'attendance_{today_date}.csv')
try:
    existing_attendance = pd.read_csv(csv_filename)
    attendance_list = existing_attendance.to_dict('records')
except FileNotFoundError:
    attendance_list = []

# Function to mark attendance
def mark_attendance(person_name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    attendance_list.append({'Name': person_name, 'Time': timestamp})
    print(f"Attendance taken for {person_name} at {timestamp}")
    # Display attendance taken message on the frame
    cv2.putText(frame, "Attendance Taken", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Variable to track the time of attendance message display
message_time = datetime.now()

while True:
    # Read a frame from the video capture
    ret, frame = video.read()

    # Detect faces using the HOG detector
    faces = hog_face_detector(frame)

    # Initialize the key variable to store the key press event
    key = cv2.waitKey(1)

    # Perform face recognition on each detected face
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Extract the face region of interest
        face_roi = frame[y:y + h, x:x + w]

        # Perform face recognition on the face region
        recognized_person = recognize_face(face_roi)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the recognized person's name above the face
        text = f"Recognized: {recognized_person}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the attendance message for a few seconds
    if len(faces) > 0 and (datetime.now() - message_time) < timedelta(seconds=3):
        cv2.putText(frame, "Attendance Taken", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Check if "B" key is pressed to mark attendance
    if key == ord('b'):
        if len(faces) > 0:
            recognized_person = recognize_face(face_roi)
            mark_attendance(recognized_person)
            message_time = datetime.now()

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if key == ord('q'):
        break

# Release the video capture
video.release()

# Destroy all windows
cv2.destroyAllWindows()

# Save attendance to CSV file
df = pd.DataFrame(attendance_list)
df.to_csv(csv_filename, index=False)
print(f"Attendance saved to {csv_filename}")



    # import cv2
    # import dlib
    # import numpy as np
    # from tensorflow.keras.models import load_model

    # # Load the trained model and person names
    # model = load_model('trained_model2.h5')
    # names = ['Haresh', 'Kushal', 'Potter', 'Shreya']

    # # Load the face detector from dlib
    # hog_face_detector = dlib.get_frontal_face_detector()

    # # Preprocess the input image
    # def preprocess_image(image):
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     resized = cv2.resize(gray, (200, 200))
    #     normalized = resized / 255.0
    #     preprocessed = normalized.reshape(1, 200, 200, 1)  # Use 1 channel for grayscale images
    #     return preprocessed

    # # Perform face recognition on an input image
    # def recognize_face(image):
    #     preprocessed_image = preprocess_image(image)
    #     predictions = model.predict(preprocessed_image)
    #     person_index = np.argmax(predictions)
    #     recognized_person = names[person_index]
    #     return recognized_person

    # # Initialize the video capture
    # video = cv2.VideoCapture(0)

    # while True:
    #     # Read a frame from the video capture
    #     ret, frame = video.read()

    #     # Detect faces using the HOG detector
    #     faces = hog_face_detector(frame)

    #     # Perform face recognition on each detected face
    #     for face in faces:
    #         x, y, w, h = face.left(), face.top(), face.width(), face.height()

    #         # Extract the face region of interest
    #         face_roi = frame[y:y + h, x:x + w]

    #         # Perform face recognition on the face region
    #         recognized_person = recognize_face(face_roi)

    #         # Draw a rectangle around the face
    #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #         # Display the recognized person's name above the face
    #         text = f"Recognized: {recognized_person}"
    #         cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #     # Display the frame
    #     cv2.imshow('Face Recognition', frame)

    #     # Break the loop when 'q' is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Release the video capture
    # video.release()

    # # Destroy all windows
    # cv2.destroyAllWindows()






# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load the trained model and person names
# model = load_model('trained_model2.h5')
# names = ['Haresh','Kushal','Ramesh']
# # names = ['haresh','jenish','kissan','kushal','monica','shreya','subibek']  

# # Load the face cascade classifier
# face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# # Preprocess the input image
# def preprocess_image(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     resized = cv2.resize(gray, (200, 200))
#     normalized = resized / 255.0
#     preprocessed = normalized.reshape(1, 200, 200, 1)
#     return preprocessed

# # Perform face recognition on an input image
# def recognize_face(image):
#     preprocessed_image = preprocess_image(image)
#     predictions = model.predict(preprocessed_image)
#     person_index = np.argmax(predictions)
#     recognized_person = names[person_index]
#     return recognized_person

# # Initialize the video capture
# video = cv2.VideoCapture(0)

# while True:
#     # Read a frame from the video capture
#     ret, frame = video.read()

#     # Convert the frame to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale frame
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Perform face recognition on each detected face
#     for (x, y, w, h) in faces:
#         # Extract the face region of interest
#         face_roi = frame[y:y + h, x:x + w]

#         # Perform face recognition on the face region
#         recognized_person = recognize_face(face_roi)

#         # Draw a rectangle around the face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Display the recognized person's name above the face
#         text = f"Recognized: {recognized_person}"
#         cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow('Face Recognition', frame)

#     # Break the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture
# video.release()

# # Destroy all windows
# cv2.destroyAllWindows()
