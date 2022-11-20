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
config_file = 'Classifier_test_values.csv'
vids = glob.glob(os.path.join(CWD_PATH, '05_Validation_Vids', '*.avi'))

csv_data = [['Video', 'Classification', 'Score']]
csv_data1 = [['Video', 'count_s', 'count_m', 'count_l', 'count_invalid', 'count_WrongD']]


def test_values(csv_file):
    if os.path.exists(os.path.join(CWD_PATH, csv_file)):
        csv_data = open(os.path.join(CWD_PATH, csv_file))
        csv_reader_obj = csv.reader(csv_data)
        info_csv = []
        for row in csv_reader_obj:
            info_csv.append(row)
        x1_ = int(info_csv[0][1])
        x_diff_ = int(info_csv[1][1])
        Average_img_ = int(info_csv[2][1])
        score_max_ = int(info_csv[3][1])
        Model_ = info_csv[4][1]
        Position_line_ = int(info_csv[5][1])
        print('loaded values:')
    else:
        x1_ = 450
        x_diff_ = 520
        Average_img_ = 2
        score_max_ = 90
        Model_ = 'DesignModel_Contur.h5'
        Position_line_ = 100
        print('preset values:')
    return x1_, x_diff_, Average_img_, score_max_, Model_, Position_line_


x1, x_diff, Average_img, score_max, Model, Position_line = test_values(config_file)
print('x1 = ' + str(x1), ': x_diff = ' + str(x_diff), ': Average_img = ' + str(Average_img),
      ': score_max = ' + str(score_max), ': Model = ' + Model)
text = ''
x1 = x1 + Position_line
x2 = x1 + x_diff
Q_Design = deque(maxlen=Average_img)
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
percent = '0'
model = load_model(os.path.join(CWD_PATH, '03_Final_Model', Model))

# cv2.namedWindow("Classification Evaluation")

def contur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    thresh = cv2.GaussianBlur(thresh, ksize=(5,5), sigmaX=0)
    # contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # cnt = contours[4]
    # im3 = cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2)

    return thresh


def write_csv(name, dataset):
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in dataset:
            writer.writerow(row)


def prepare_folder():
    filelist = glob.glob(os.path.join(CWD_PATH, '06_Validation_results', '*.jpg'))
    for f in filelist:
        os.remove(f)


def add_csv(file, data):
    with open(os.path.join(CWD_PATH, file), 'a', newline='') as file_to_write:
        writer = csv.writer(file_to_write)
        writer.writerow(data)


def mySort(e):
  return e

csv_data = [['Video', 'Classification', 'Score']]
csv_data1 = ['Video', 'count_s', 'count_m', 'count_l', 'count_invalid', 'count_WrongD']
add_csv('06_Validation_results/Summary.csv', csv_data1)
prepare_folder()
result = 'empty'
for vid in vids: #######
    count_s = 0
    count_m = 0
    count_l = 0
    count_Invalid = 0
    count_WrongD = 0
    summary = [['S', 0], ['M', 0], ['L', 0]]
    # for i in range(10, 15):
    #     x1 = (i + 1) * 50
    #     x2 = x1 + 500
    #     print(x1, x2)
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

        frame = frame[210:510, x1:x2]
        #print(frame.shape)
        save = frame.copy()

        frame = contur(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        classify_img = cv2.resize(frame, (250, 160))
        classify_img = img_to_array(classify_img)
        classify_img = np.array(classify_img, dtype="float")
        classify_img = classify_img.astype('float32') / classify_img.max()
        classify_img = np.reshape(classify_img, (-1, 160, 250, 3))

        classifier = model.predict(classify_img,batch_size=1)

        Q_Design.append(classifier)
        classifier_0 = np.array(Q_Design).mean(axis=0)
        prediction = np.argmax(classifier_0)
        percent = classifier[0][prediction] * 100

        # prediction = np.argmax(classifier)
        # percent = classifier[0][prediction]*100
        if len(Q_Design) == Average_img:
            if prediction == 0 and percent >= score_max:# and percent>95.00:
                result = "Design:S\n"
                count_s += 1
                summary[0][1] += 1
                pfad = os.path.join(CWD_PATH, '06_Validation_results', 'S', v_name[0] + "-" + str(count_s) + ".jpg")
                # cv2.imwrite(pfad, save)
                # print(result, percent)
                add_csv('06_Validation_results/Summary_detail.csv', [v_name[0], "Design:S", percent])
                # break
            elif prediction == 1 and percent >= score_max:
                result = "Design:M\n"
                count_m += 1
                summary[1][1] += 1
                pfad = os.path.join(CWD_PATH, '06_Validation_results', 'M', v_name[0] + "-" + str(count_m) + ".jpg")
                # cv2.imwrite(pfad, save)
                # print(result, percent)
                add_csv('06_Validation_results/Summary_detail.csv', [v_name[0], "Design:M", percent])
                # break
            elif prediction == 2 and percent >= score_max:
                result = "Design:L\n"
                count_l += 1
                summary[2][1] += 1
                pfad = os.path.join(CWD_PATH, '06_Validation_results', 'L', v_name[0] + "-" + str(count_l) + ".jpg")
                # cv2.imwrite(pfad, save)
                # print(result, percent)
                add_csv('06_Validation_results/Summary_detail.csv', [v_name[0], "Design:L", percent])
                # break
            elif prediction == 3:
                result = "Design:Invalid\n"
                count_Invalid += 1
           # print(text)
        else:
            print(' ')
            #add_csv('06_Validation_results/Summary_detail.csv', [v_name[0], result, percent])
            #print(v_name[0], result, percent)
        count += 1
        cv2.putText(disp, result, (70, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255),
                    lineType=cv2.LINE_AA)
        cv2.line(disp, (Position_line, 0), (Position_line, 720), (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Image", disp)
        cv2.imshow("Detail", frame)
        key = cv2.waitKey(1) & 0xFF
        if (count > (video_length - 1)):
            cap.release()
            cv2.destroyAllWindows()
            break

    # csv_data1 += [[v_name[0], count_S, count_m, count_l, count_Invalid, count_WrongD]]
    # print(summary)
    summary.sort(reverse=True, key=lambda summary: summary[1])
    # print(summary)
    # print(summary[0][0])
    add_csv('06_Validation_results/Summary.csv', [v_name[0], count_s, count_m, count_l, count_Invalid, count_WrongD,summary[0][0]])
#write_csv('06_Validation_results/Detail.csv', csv_data)
#write_csv('06_Validation_results/Summary.csv', csv_data1)

#cap.release()

cv2.destroyAllWindows()
