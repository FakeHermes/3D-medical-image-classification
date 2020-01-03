from mylib.dataloader.dataset import ClfDataset, get_balanced_loader, get_loader, get_x_test
from mylib.models import densesharp, metrics, losses
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import tensorflow as tf
import keras
import os
import _pickle as pickle
import numpy as np
import codecs
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'
def main():
    #读取保存在磁盘上的模型

    test=get_x_test()
    print(np.shape(test))
    
    with open('result/p3.pickle','rb') as fr:
        new_cnns1 = pickle.load(fr)
    model=new_cnns1
    y=model.predict(test)
    print('p3')
    for i in range(117):
        print(y[i,1])

    with open('result/p4.pickle','rb') as fr:
        new_cnns1 = pickle.load(fr)
    model=new_cnns1
    y=model.predict(test)
    print('p4')
    for i in range(117):
        print(y[i,1])
    
    with open('result/p5.pickle','rb') as fr:
        new_cnns1 = pickle.load(fr)
    model=new_cnns1
    y=model.predict(test)
    print('p5')
    for i in range(117):
        print(y[i,1])


if __name__ == '__main__':
    main()
