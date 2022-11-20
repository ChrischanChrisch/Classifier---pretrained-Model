import cv2
import numpy as np
import os
import tensorflow as tf
import glob
import csv
from collections import deque

from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array

CWD_PATH = os.getcwd()
model = load_model(os.path.join(CWD_PATH, '03_Final_Model', 'Stent_Type_Final_Model_3_classes.h5'))
Average_img = 5
vids = glob.glob(os.path.join(CWD_PATH, '05_Validation_Vids', '*.avi'))

csv_data = [['Video', 'Classification', 'Score']]
csv_data1 = [['Video', 'count_s', 'count_invalid', 'count_WrongD']]

Q_Design = deque(maxlen=Average_img)
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
#cv2.namedWindow("Classification Evaluation")

def write_csv(name, dataset):
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in dataset:
            writer.writerow(row)

def prepare_folder():
    filelist = glob.glob(os.path.join(CWD_PATH, '06_Validation_results', '*.jpg'))
    for f in filelist:
        os.remove(f)


for vid in vids: #######
    count_s = 0
    count_Invalid = 0
    count_WrongD = 0
    for i in range(7, 10):
        x1 = (i + 1) * 50
        x2 = x1 + 330
        print(x1, x2)
        cap = cv2.VideoCapture(vid)
        vid_name = vid.split(os.sep)
        v_name = vid_name[len(vid_name) - 1].split('.')
        print(v_name[0])
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        count = 0
        Q_Design.clear()
        while (True):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            disp = frame.copy()
            #frame = frame[300:-200, 450:-500]

            frame = frame[250:470, x1:x2]
            #print(frame.shape)
            save = frame.copy()

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            classify_img = cv2.resize(frame, (250, 160))
            classify_img = img_to_array(classify_img)
            classify_img = np.array(classify_img, dtype="float")
            classify_img = classify_img.astype('float32') / classify_img.max()
            classify_img = np.reshape(classify_img, (-1, 160, 250, 3))

            classifier = model.predict(classify_img, batch_size=1)
            Q_Design.append(classifier)
            classifier_0 = np.array(Q_Design).mean(axis=0)
            # prediction = np.argmax(classifier)
            # percent = classifier[0][prediction] * 100

            prediction = np.argmax(classifier_0)
            percent = classifier_0[0][prediction]*100
            if len(Q_Design) == Average_img:
                if prediction == 0 and percent > 95.00:  # and percent >= score_max:# and percent>95.00:
                    result = "Design:S\n"
                    count_s += 1
                    pfad = os.path.join(CWD_PATH, '06_Validation_results', v_name[0] + "-" + str(x1) + "-" + str(count_s) + ".jpg")
                    cv2.imwrite(pfad, save)
                    print(v_name[0], result, percent)
                    break
                elif prediction == 1:  # and percent >= score_max:
                    result = "Design:Invalid\n"
                    count_Invalid += 1
                elif prediction == 2:
                    result = "Design:Wrong Diameter\n"
                    count_WrongD += 1
                # print(text)
                # csv_data += [[v_name[0], result, percent]]
                # print(v_name[0], result, percent)

                cv2.putText(disp, result, (70, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255),
                            lineType=cv2.LINE_AA)
            else:
                print('_')

            count += 1
            cv2.imshow("Image", disp)
            cv2.imshow("Detail", frame)
            key = cv2.waitKey(1) & 0xFF
            if (count > (video_length - 1)):
                print('video ende')
                cap.release()
                cv2.destroyAllWindows()
                break

    csv_data1 += [[v_name[0], count_s, count_Invalid, count_WrongD]]
    # write_csv('06_Validation_results/Detail.csv', csv_data)
    write_csv('06_Validation_results/Summary.csv', csv_data1)

cap.release()

cv2.destroyAllWindows()
