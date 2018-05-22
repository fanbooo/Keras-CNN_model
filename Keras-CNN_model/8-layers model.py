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
    im = np.ones((len(image_list),224,224,3))
    label = np.ones((len(image_list),1))
    for i in range(len(label_list)):
        img = Image.open(image_list[i])
        imgg = img.resize((224,224))
        arr = np.asarray(imgg, dtype="float32")
        im[i,:,:,:] = arr
        label[i] = label_list[i]

    train, test, train_label, test_label = train_test_split(im, label, test_size=0.2, random_state=42)
#    =========================shuffle=============== 
    return train, test, train_label, test_label

def model_train(train, test, train_label, test_label):
    model = Sequential()
    model.add(Convolution2D(4, 5, 5,input_shape=(224, 224,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #第二个卷积层，8个卷积核，每个卷积核大小3*3。
    #激活函数用tanh
    #采用maxpooling，poolsize为(2,2)
    model.add(Convolution2D(8, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #第三个卷积层，16个卷积核，每个卷积核大小3*3
    #激活函数用tanh
    #采用maxpooling，poolsize为(2,2)
    model.add(Convolution2D(16, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #全连接层，先将前一层输出的二维特征图flatten为一维的。
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #sigmoid分类，输出是2类别
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])                   
    
  
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


