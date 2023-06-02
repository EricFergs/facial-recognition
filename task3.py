import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps


def get_className(classNo):
    if classNo == 0:
        return "Eric"
    elif classNo == 1:
        return "Bryan"
def webcams(num_webcams):
    facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_COMPLEX
    model = load_model('keras_model.h5', compile=False)
    captureList = []
    for cams in range(num_webcams):
        cap = cv2.VideoCapture(cams)
        cap.set(3, 640)
        cap.set(4, 480)
        captureList.append(cap)

    while True:
        frames = []
        for cap in captureList:
            success, frame = cap.read()
            frames.append(frame)
        for i in frames:
            faces = facedetect.detectMultiScale(i, 1.3, 5)
            for x, y, w, h in faces:
                crop_img = i[y:y+h, x:x+h]
                img = cv2.resize(crop_img, (224, 224))
                img = img.reshape(1, 224, 224, 3)

                image_array = np.asarray(img)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                prediction = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)  # new array

                prediction = model.predict(normalized_image_array)

                classIndex = np.argmax(prediction)
                probabilityValue = np.amax(prediction)

                if classIndex == 0:
                    cv2.rectangle(i, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.rectangle(i, (x, y-40), (x+w, y), (0, 255, 0), -2)
                    cv2.putText(i, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                elif classIndex == 1:
                    cv2.rectangle(i, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.rectangle(i, (x, y-40), (x+w, y), (0, 255, 0), -2)
                    cv2.putText(i, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(i, str(round(probabilityValue*100, 2))+"%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        for i in range(len(frames)):
            cv2.imshow(f'Camera {i}', frames[i])
        exit = cv2.waitKey(1) & 0xff
        if exit == ord('d'):
            break
    for cap in captureList:
        cap.release()
    cv2.destroyAllWindows()

webcams(2)
