import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import imutils
import matplotlib.pyplot as plt

import cv2
#import tensorflow as tf
import numpy as np

model = load_model(r'C:\Tensorflow2\workspace\Classifier_2\Final_Model\Model.h5')
#print('Loaded model from the disk')

cam = cv2.VideoCapture(0)

cv2.namedWindow("Classification Evaluation")

img_counter = 1
j=0
text=''
while (cam.isOpened()):
    j=str(j)
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame=cv2.resize(frame,(800,650)) #for s and l
    frame=cv2.resize(frame,(400,250))
    #cv2.imshow('Vid',frame)
    #cv2.waitKey(0)
    #frame = frame[1:-1, 80:-1]
    frame=frame[40:-40, 90:-50] #for s and l


    classify_img = cv2.resize(frame, (250, 160))
    classify_img = img_to_array(classify_img)
    classify_img = np.array(classify_img, dtype="float")
    classify_img = classify_img.astype('float32') / classify_img.max()
    classify_img = np.reshape(classify_img, (-1,160, 250, 3))


    classifier = model.predict(classify_img,batch_size=1)


    prediction = np.argmax(classifier)
    percent = classifier[0][prediction]*100
    #
    if prediction == 0 and percent>99.0:
         percent=str(percent)
         text = 'S'
    elif prediction == 1:# and percent>95.00:
         percent=str(percent)
         text = 'Invalid'
    elif prediction == 2:# and percent>95.00:
        percent=str(percent)
        text = 'Wrong Diameter'
   # elif prediction == 3 and percent>95.00:
    #     percent=str(percent)
    #     text = 'Invalid'


    cv2.putText(frame, text, (70, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255),
                lineType=cv2.LINE_AA)
    #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # frame=(frame-frame.min())/(frame.max()-frame.min())
    j=int(j)
    j=j+1
    cv2.imshow("Image", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

cam.release()

cv2.destroyAllWindows()

#model=load_model(r'C:\Users\csadmin\PycharmProjects\April_DAE\Stent_Type_Classification\02_Saved_Models\02_Aug26_conv2_block1_4_classes\Resnet_conv2_block1_False.91-0.00.h5')
#
# img=cv2.imread(r'C:\Users\csadmin\PycharmProjects\April_DAE\Stent_Type_Classification\3_.jpg')
# img=img[200:-200,300:-500]
# frame = cv2.resize(img, (250, 160))
# classify_img = img_to_array(frame)
# classify_img = np.array(classify_img, dtype="float")
# classify_img = classify_img.astype('float32') / classify_img.max()
# classify_img = np.reshape(classify_img, (-1, 160, 250, 3))
# classifier = model.predict(classify_img, batch_size=1)
#
# prediction = np.argmax(classifier)
#
# if prediction==0:
#     text='T6L'
# elif prediction==1:
#     text='T6S'
# elif prediction==2:
#     text='Invalid'
#
# cv2.imshow('Img',img)
# print(text)
# cv2.putText(img, text, (100, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
# cv2.waitKey(0)
# cv2.show()
# cam = cv2.VideoCapture(r'C:\Users\csadmin\PycharmProjects\April_DAE\Stent_Type_Classification\05_Testing_videos\t6s\S_1.flv')
# #cam = cv2.VideoCapture(0)
#
# cv2.namedWindow("Stent Type")
# img_counter = 0
#
# while (cam.isOpened()):
#      ret, frame = cam.read()
#      #if ret:
#      #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#      #frame = frame[10:-80, 20:-80]#to remove the black borders
#      #frame_1=cv2.resize(frame,(1280,720))#to make it compatible with the images taken
#      #frame_1 = frame_1[100:-100, 200:-200]#find the ROI for stent type
#      #disp = frame_1.copy()#display the ROI
#
# #      frame = cv2.resize(frame, (250, 160))
# #      classify_img = img_to_array(frame)
# #      classify_img = np.array(classify_img, dtype="float")
# #      classify_img = classify_img.astype('float32') / classify_img.max()
# #      classify_img = np.reshape(classify_img, (-1, 160, 250, 3))
# #      classifier = model.predict(classify_img, batch_size=1)
# #
# #      prediction = np.argmax(classifier)
# #
# #      if prediction == 0:
# #          text = 'T6L'
# #          cv2.imshow('Frame',disp)
# #          cv2.waitKey(10)
# #      elif prediction == 1:
# #          text = 'T6M'
# #          cv2.imshow('Frame', disp)
# #          cv2.waitKey(10)
# #      elif prediction == 2:
# #          text = 'T6S'
# #          cv2.imshow('Frame', disp)
# #          cv2.waitKey(10)
# #      elif prediction==3:
# #          text='Invalid'
# #      elif prediction==4:
# #          text='Wrong Diameter'
# #
# #
# # #     frame = cv2.resize(frame, (250, 160))
# #      cv2.putText(frame_1, text, (80, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
# #      cv2.imshow("test", frame_1)
#
#      if not ret:
#          break
#      k = cv2.waitKey(1)
# #
#      if k % 256 == 27:
#          # ESC pressed
#          print("Escape hit, closing...")
#          break
#      elif k % 256 == 32:
#          # SPACE pressed
#          img_name = "No_{}.jpg".format(img_counter)
#          cv2.imwrite(img_name, frame)
#          print("{} written!".format(img_name))
#          img_counter += 1
# #
# cam.release()
# #
# cv2.destroyAllWindows()