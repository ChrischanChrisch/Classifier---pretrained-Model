import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf

from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.models import load_model

model = load_model(r'C:\KI\D\Stent_Type_BES\03_Final_Model\DesignModel.h5')
Text = ''
x1 = 550
x2 = 1000
y1 = 200
y2 = 450

for imag in glob.glob(r"C:\KI\D\Stent_Type_BES\New_Pics/*.jpg"):
    print(imag)

    frame = cv2.imread(imag)
    frame = cv2.resize(frame, (1280, 720))
    disp = frame.copy()
    frame = frame[y1:y2, x1:x2]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    classify_img = cv2.resize(frame, (250, 160))
    classify_img = img_to_array(classify_img)
    classify_img = np.array(classify_img, dtype="float")
    classify_img = classify_img.astype('float32') / classify_img.max()
    classify_img = np.reshape(classify_img, (-1, 160, 250, 3))

    classifier = model.predict(classify_img, batch_size=1)
    prediction = np.argmax(classifier)
    percent = classifier[0][prediction]*100

    if prediction == 0:
        Text = 'Design: S '
        print(Text, percent)
    elif prediction == 1:
        Text = 'Design: Invalid '
        print(Text, percent)
    elif prediction == 2:
        Text = 'Design: Wrong '
        print(Text, percent)
    elif prediction == 3:
        Text = 'Design: M'
        print(Text, percent)
    elif prediction == 4:
        Text = 'Design: L '
        print(Text, percent)

    cv2.putText(disp, Text, (70, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255),
                lineType=cv2.LINE_AA)
    cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.imshow("Image", disp)

    k = cv2.waitKey(0)
cv2.destroyAllWindows()