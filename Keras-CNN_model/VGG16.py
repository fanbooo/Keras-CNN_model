# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:36:23 2018

@author: fanbooo
"""
import os  # 处理字符串路径
import glob  # 查找文件
import keras
from keras.models import Sequential  # 导入Sequential模型
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D as Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
import numpy as np
from keras import layers, optimizers
from pandas import Series, DataFrame
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.layers.pooling import AveragePooling2D
#from six.moves import range
from PIL import Image
from sklearn.cross_validation import train_test_split
import numpy as np
#读取文件夹train下的42000张图片，图片为彩色图，所以为3通道，
#如果是将彩色图作为输入,图像大小224*224

def get_files(file_dir):
    """
    scan the local file_dir to assemble image_list and label_list
    """
    img_list = []
    label_list = []

    for train_class in os.listdir(file_dir):
        for pic in os.listdir(file_dir + "\\" + train_class):
            img_list.append(file_dir + "\\" + train_class + '\\' + pic)
            label_list.append(train_class)
    temp = np.array([img_list, label_list])
    temp = temp.transpose()
    # shuffle the samples
    np.random.shuffle(temp)
    # after transpose, images is in dimension 0 and label in dimension 1
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    im = np.ones((len(image_list),227,227,3))
    label = np.ones((len(image_list),2))
    for i in range(len(label_list)):
        img = Image.open(image_list[i])
        imgg = img.resize((227,227))
        arr = np.asarray(imgg, dtype="float32")
        im[i,:,:,:] = arr
        if label_list[i] == 0:
            label[i,1] = 0
        else:
            label[i,0] = 0

    train, test, train_label, test_label = train_test_split(im, label, test_size=0.2, random_state=42)
#    valid, test, valid_label, test_label =train_test_split(val, val_label, test_size=0.5, random_state=42)
#    =========================shuffle=============== 
#    return train, valid, test, train_label, valid_label, test_label
    return train, test, train_label, test_label

def model_train(train, test, train_label, test_label):
    model = Sequential()  
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Flatten())  
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(4096,activation='relu'))  
    model.add(Dropout(0.5))  
    model.add(Dense(1000,activation='softmax')) 
    
    
    
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer = sgd,metrics=['accuracy'])                   
    
  
    train = train.astype('float32')
    test = test.astype('float32')
    train /= 255
    test /= 255
    
    
    model.fit(train, train_label,
             nb_epoch=100, batch_size=64,
             validation_data=(test,test_label))
    
    return model

def model_evaluate(model,test,test_label):
    scores = model.evaluate(test, test_label, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
def main():
    file_dir = "your path"
    train, test, train_label, test_label = get_files(file_dir)
    model = model_train(train, test, train_label, test_label)
    model_evaluate(model,test,test_label)
    Y = model.predict(test)
    return test,Y


