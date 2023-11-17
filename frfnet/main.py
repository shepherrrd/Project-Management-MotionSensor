from PIL import Image

from tensorflow.keras.models import load_model
import numpy as np
from numpy import asarray
from numpy import expand_dims
from keras_facenet import FaceNet
import pickle
import cv2
def onlychar(input_string):
    """
    Removes non-alphabetic characters from a string and returns the result.

    Parameters:
    input_string (str): The input string.

    Returns:
    str: The input string with only alphabetic characters.
    """
    result_string = ''.join(char for char in input_string if char.isalpha())
    return result_string
#HaarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()

myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()

cap = cv2.VideoCapture(0)

while(1):
    _, gbr1 = cap.read()
    
    wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 4)
    
    if len(wajah) > 0:
        x1, y1, width, height = wajah[0] 
        x1, y1, x2, y2 = x1, y1, x1 + width, y1 + height  # Updated this line
        
        x1, y1 = abs(x1), abs(y1)
    
        gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
        gbr = Image.fromarray(gbr)                  
        gbr_array = asarray(gbr)
        
        face = gbr_array[y1:y2, x1:x2]                        
        
        face = Image.fromarray(face)                       
        face = face.resize((160, 160))
        face = asarray(face)
        face = expand_dims(face, axis=0)
        signature = MyFaceNet.embeddings(face)
        
        min_dist = 100
        identity = 'Unknown'  # Default to 'Unknown'
        
        for key, value in database.items():
            dist = np.linalg.norm(value - signature)
            if dist < min_dist:
                min_dist = dist
                identity = onlychar(key)
        
        # Check if the distance is above a certain threshold
        threshold = 1.0  # You can adjust this threshold based on your needs
        if min_dist > threshold:
            identity = 'Unknown'
            print(signature)
            color = (0, 0, 255)  # Red for 'Unknown'
        else:
            color = (0, 255, 0)  # Green for known identities
        
        cv2.putText(gbr1, identity, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(gbr1, (x1, y1), (x2, y2), color, 2)
    
    cv2.imshow('res', gbr1)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()
