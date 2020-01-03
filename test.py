from mylib.dataloader.dataset import get_x_test
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf
import keras
import os
import _pickle as pickle
import numpy as np
import codecs
import csv
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'
def main():
    
    test=get_x_test()
 
    with open('result/p3.pickle','rb') as fr:
        new_cnns1 = pickle.load(fr)
    model=new_cnns1
    y3=model.predict(test)
    print('p3')

    with open('result/p4.pickle','rb') as fr:
        new_cnns1 = pickle.load(fr)
    model=new_cnns1
    y4=model.predict(test)
    print('p4')
    
    with open('result/p5.pickle','rb') as fr:
        new_cnns1 = pickle.load(fr)
    model=new_cnns1
    y5=model.predict(test)
    print('p5')
    
    #直接读取excel数据
    list_path=pd.read_csv(os.path.join('result/5_avg.csv'))
    y1=list_path['p1']
    y2=list_path['sample']
    y=np.zeros(117)
    for i in range(117):
        y[i]=(y1[i]+y2[i]+y3[i,1]+y4[i,1]+y5[i,0])/5 
        print(y[i])
    


if __name__ == '__main__':
    main()
