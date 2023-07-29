import cv2
import dlib
import os

video = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()

faces_data = []
i = 0

name = input("Enter Your Name: ")

# Create a folder to store images if it doesn't exist
folder_path = os.path.join("data", "images", name)
os.makedirs(folder_path, exist_ok=True)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the HOG detector
    faces = hog_face_detector(gray)

    # Perform face processing only if at least one face is detected
    if len(faces) > 0:
        face = faces[0]  # Take the first detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        if w > 0 and h > 0:  # Check if the detected face has a valid size
            crop_img = frame[y : y + h, x : x + w, :]
            if crop_img.size != 0:  # Check if the crop_img is not empty
                resized_img = cv2.resize(crop_img, (200, 200))  # Resize to 200x200 pixels
                gray_img = cv2.cvtColor(
                    resized_img, cv2.COLOR_BGR2GRAY
                )  # Convert to grayscale
                if len(faces_data) <= 100 and i % 10 == 0:
                    faces_data.append(resized_img)
                    # Save the grayscale image in the provided folder
                    cv2.imwrite(
                        os.path.join(folder_path, f"image_{len(faces_data)}.jpg"),
                        resized_img,
                    )
                i += 1
                cv2.putText(
                    frame,
                    str(len(faces_data)),
                    (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (50, 50, 255),
                    1,
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

    cv2.imshow("Capturing...", frame)
    k = cv2.waitKey(1)
    if k == ord("q") or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()




# import cv2
# import dlib
# import os

# video = cv2.VideoCapture(0)
# hog_face_detector = dlib.get_frontal_face_detector()

# faces_data = []
# i = 0

# name = input("Enter Your Name: ")

# # Create a folder to store images if it doesn't exist
# folder_path = os.path.join("data", "images", name)
# os.makedirs(folder_path, exist_ok=True)

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces using the HOG detector
#     faces = hog_face_detector(gray)

#     for face in faces:
#         x, y, w, h = face.left(), face.top(), face.width(), face.height()
#         if w > 0 and h > 0:  # Check if the detected face has a valid size
#             crop_img = frame[y : y + h, x : x + w, :]
#             resized_img = cv2.resize(crop_img, (200, 200))  # Resize to 200x200 pixels
#             gray_img = cv2.cvtColor(
#                 resized_img, cv2.COLOR_BGR2GRAY
#             )  # Convert to grayscale
#             if len(faces_data) <= 100 and i % 10 == 0:
#                 faces_data.append(resized_img)
#                 # Save the grayscale image in the provided folder
#                 cv2.imwrite(
#                     os.path.join(folder_path, f"image_{len(faces_data)}.jpg"),
#                     resized_img,
#                 )
#             i += 1
#             cv2.putText(
#                 frame,
#                 str(len(faces_data)),
#                 (50, 50),
#                 cv2.FONT_HERSHEY_COMPLEX,
#                 1,
#                 (50, 50, 255),
#                 1,
#             )
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

#     cv2.imshow("Capturing...", frame)
#     k = cv2.waitKey(1)
#     if k == ord("q") or len(faces_data) == 100:
#         break

# video.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import os

# video = cv2.VideoCapture(0)
# facesdetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# faces_data = []
# i = 0

# name = input("Enter Your Name: ")

# # Create a folder to store images if it doesn't exist
# folder_path = os.path.join('data', 'images', name)
# os.makedirs(folder_path, exist_ok=True)

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     faces = facesdetect.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w, :]
#         resized_img = cv2.resize(crop_img, (200, 200))  # Resize to 200x200 pixels
#         gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#         if len(faces_data) <= 100 and i % 10 == 0:
#             faces_data.append(resized_img)
#             # Save the grayscale image in the provided folder
#             cv2.imwrite(os.path.join(folder_path, f'image_{len(faces_data)}.jpg'), resized_img)
#         i += 1
#         cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

#     cv2.imshow("Capturing...", frame)
#     k = cv2.waitKey(1)
#     if k == ord('q') or len(faces_data) == 100:
#         break

# video.release()
# cv2.destroyAllWindows()
