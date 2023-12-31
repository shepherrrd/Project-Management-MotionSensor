import threading
import cv2
from deepface import DeepFace
import requests
## to setup run pip install deepface opencv-python tensorflow
## to run it use python3 main.py
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
#dd

face_match = False
img = cv2.imread("face1.jpg")
img = cv2.resize(img, (640, 480))
siamese_model = tf.keras.models.load_model('./FaceRecognition/siamesemodel.h5')
def check_face(frame):
    global face_match
    try:
        result = DeepFace.verify(frame, img.copy())
        if result["verified"]:
            face_match = True
        else:
            requests.post('http://localhost:3000/ping', data={'status': 'unauthorized', 'img' : frame})
            face_match = False
    except ValueError:
        face_match = False

while True:
    ret,frame = cap.read()

    if ret:
        if counter %30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter +=1
    if face_match:
        cv2.putText(frame,"Match!",(20,40),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    else:
        cv2.putText(frame,"No Match!",(20,40),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
    cv2.imshow("video",frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
