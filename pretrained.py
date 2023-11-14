import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import requests


model = MobileNetV2(weights='imagenet')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def classify_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions

img_path = 'face.jpg'
predictions = classify_image(img_path)
print(predictions)




# Load the pre-trained FaceNet model
face_net_model = load_model('path/to/your/facenet_model.h5')

# Threshold for face verification
verification_threshold = 0.7

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()

    # Assume face detection logic is implemented here
    # For simplicity, you can use a pre-trained face detection model or Haarcascades

    # Extract face region
    face_roi = frame[y:y+h, x:x+w]

    # Preprocess the face image for FaceNet
    face_array = cv2.resize(face_roi, (160, 160))
    face_array = np.expand_dims(face_array, axis=0)
    face_array = preprocess_input(face_array)

    # Use FaceNet for face verification
    embeddings = face_net_model.predict(face_array)

    # Assuming authorized_face_embeddings is the embedding of your authorized face
    authorized_face_embeddings = ...

    # Calculate L2 distance for face verification
    distance = np.linalg.norm(embeddings - authorized_face_embeddings)

    # Display the frame with rectangle around the face
    if distance < verification_threshold:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Authorized face (green)
    else:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Unauthorized face (red)

        # Send a POST request to the desired endpoint for unauthorized face
        requests.post('http://localhost:3000/ping', data={'status': 'unauthorized'})

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
