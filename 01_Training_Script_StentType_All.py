import datetime
import os
import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd

from skimage import io
from keras.models import Model
from keras.layers import Conv2D,Dense,Flatten,MaxPool2D,Activation,Dropout,UpSampling2D,add,Conv2DTranspose,concatenate,multiply,GlobalMaxPooling2D,GlobalAveragePooling2D,Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard
from keras.applications.vgg16 import VGG16
#from keras.applications.resnet import ResNet50
#from keras.applications.inception_v3 import InceptionV3

batch_size = 32
epochs = 100
learn_rate = 0.0001
#reg = l2(0.0005)

i = 0
path = r'C:\Tensorflow2\workspace\Classifier_2\01_Train_Data' #D:\Christian_KI_Projekt\CNN\Categories'
subfolders = [f.name for f in os.scandir(path) if f.is_dir()]
print('Anzahl Kategorien: ', len(subfolders))
print(subfolders)

categorie_images = []
categorie_label = []
categorie_name = []
#categorie = []
#label = []
for i in range(0, len(subfolders)):
    categorie = []
    label = []
    for img in os.listdir(path+'\\'+subfolders[i]):
        #print(img)
        img_1 = io.imread(path+'\\'+subfolders[i]+'\\'+img)
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
        img_1 = cv2.resize(img_1, (250, 160))
        #img_1 = resize(img_1, (img_1.shape[0] / 4, img_1.shape[1] / 4))
        #img_1 = resize(img_1, (img_1.shape[0], img_1.shape[1]))
        categorie.append(img_1)
        label.append(i)
        shape = img_1.shape
    categorie_images.append(categorie)
    categorie_label.append(label)
    categorie_name.append(subfolders[i])

for i in range(0, len(categorie_images)):
    print('No. in: ', categorie_name[i], ' are: ', len(categorie_images[i]))

all_together = categorie_images[0]+categorie_images[1]+categorie_images[2]+categorie_images[3]  #  +categorie_images[4]
print('All_together are:', len(all_together))
train_X = np.array(all_together)

labels = categorie_label[0]+categorie_label[1]+categorie_label[2]+categorie_label[3]  #  +categorie_label[4]

x_train = train_X.astype('float32') / train_X.max()
x_train = np.reshape(x_train, (len(x_train), 160, 250, 3))

train_labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

train_x, train_y = shuffle(x_train, train_labels)
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.05, shuffle=True)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.05, shuffle=True)
val_x, val_y = shuffle(val_x, val_y)

################################################
#base_model = ResNet50(weights= r'C:\KI\D\Stent_Type_BES\04_Pretrained_Model\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape= (160,200,3))
#x = base_model.get_layer('conv3_block1_out').output
#x.trainable = False
# # # # # conv_8 = Conv2D(32,kernel_size=3,padding='same',activation='relu',name='conv2d_2')(x)
# # # # # conv_9 = Conv2D(128,kernel_size=3,padding='same',activation='relu',name='conv2d_3')(conv_8)
#global_pool = GlobalMaxPooling2D()(conv_9)
#hidden_1 = Dense(100,activation='relu')(global_pool)
#drop_2 = Dropout(0.5)(hidden_1)
#output_1 = Dense(5, activation='softmax')(drop_2)
#model = Model(inputs=base_model.input,outputs=base_model.output)
#################################

# # ################################################
# #21.08.2020
base_model = VGG16(weights=r'C:\Tensorflow2\workspace\Classifier_2\04_Pretrained_Model\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(160, 250, 3))
x = base_model.get_layer('block3_conv3').output
x.trainable = False
# # # # # #inputs=Input(shape=(160,200,3))
# # # # # #x=base_model(inputs)
# # # # # #visible=Input(shape=(160,200,3))
# # # # # #conv_8=Conv2D(128,kernel_size=3,padding='same',activation='relu',name='conv2d_2')(x)
# # # # # #conv_9=Conv2D(128,kernel_size=3,padding='same',activation='relu',name='conv2d_3')(conv_8)
global_pool = GlobalAveragePooling2D()(x)
hidden_1 = Dense(50, activation='relu')(global_pool)
drop_2 = Dropout(0.5)(hidden_1)
output_1 = Dense(4, activation='softmax')(drop_2)
model = Model(inputs=base_model.input, outputs=output_1)


print(model.summary())

check_point = keras.callbacks.ModelCheckpoint(filepath=r'C:\Tensorflow2\workspace\Classifier_2\03_Final_Model\DesignModel_Contur.h5',monitor='val_loss',save_best_only=True)
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [check_point, tensor_board]

opt = Adam(lr=learn_rate, decay=0.000009)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])

history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1, validation_data=(val_x, val_y))

score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
print("Evaluation Accuracy %s: %.2f%%" %(model.metrics_names[1], score[1]*100))
#model.save(r'C:\Users\csadmin\PycharmProjects\April_DAE\End_Classification\07_Training_Data_Close\05_May_26_Classification_with_Categories\03_Training_Scripts\08_July15_Resnet_bs_16_conv3_block3\July15_Last_epoch.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

