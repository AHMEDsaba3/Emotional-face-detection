from keras.models import model_from_json
from keras.preprocessing.image import load_img
import numpy as np
import cv2


# Load json and create model
json_file = open(
    'D:/semester 7/digital image/project/models/model.ult/emotiondetector.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
# load weights into new model
emotion_model.load_weights("D:/semester 7/digital image/project/models/model.ult/emotiondetector.h5")
print("Loaded model from disk")


label = ['angry','disgust','fear','happy','neutral','sad','surprise']

#preprocess an image before making predictions.
def ef(image):
    img = load_img(image, color_mode='grayscale')
    feature = np.array(img)
    # Resize the image to 48x48 pixels
    feature = cv2.resize(feature, (48, 48))
    # Reshape the array
    feature = feature.reshape(1, 48, 48, 1) 
    return feature / 255.0

image_path = 'D:/semester 7/digital image/project/external test/ddf70db78b0f3ee2388cedbdd66f93e0.jpg'
img = ef(image_path)
pred = emotion_model.predict(img)
emotion_model = label[pred.argmax()]
print("model prediction is ",emotion_model)

def draw_square_with_label(image_path):
    img = cv2.imread(image_path)
    img=cv2.resize(img, (800,600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml.')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(5, 5))
    #print(type(faces))
    if faces is not tuple() :
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, emotion_model, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    else:
        cv2.rectangle(img, (40, 40), (40+690, 40+550), (255, 0, 0), 2)
        cv2.putText(img, emotion_model, (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        
    cv2.imshow('Image with Square and Label on Face', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Draw squares on faces, make emotion prediction, and write labels
draw_square_with_label(image_path)






