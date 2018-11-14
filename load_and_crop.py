import os
import sys
import pandas
import numpy as np
import argparse
import pandas as pd
import csv
import cv2
from keras.models import Sequential
from keras.layers import Input,MaxPooling2D,Convolution2D, BatchNormalization, Dense, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint
#from keras.preprocessing import Image
from keras.preprocessing.image import ImageDataGenerator,DirectoryIterator


def readCSV(route,filename):
    data=[]

    ## usando pandas
    df=pd.read_csv(filename,delimiter=';') 
    data=np.asarray(df)        
    for i in data:
        i[0]=route+'/'+i[0]
    #print (data)

    return np.asarray(data) 


def readAllCSVs(route):
    data=[]
    ret=[]
    #print('entering '+route)            
    for i in os.listdir(route):
        filename=route+'/'+str(i)
        if os.path.isdir(filename):
            ret=readAllCSVs(filename)
            if(len(ret)>0):
                data.append(np.asarray(ret))           
        else:            
            #print('loading '+filename)
            aux=str(i)
            aux=aux.split('.')
            aux=aux[len(aux)-1]
            if aux=='csv' or aux=='CSV':
                return readCSV(route,filename)                                                
    data=np.concatenate(data)
    data=np.asarray(data)       
    return data        

import cv2
#import keras.preprocessing.Image
def load_and_preprocess(filename,x0,y0,x1,y1,size_x,size_y,save_to_dir=''):
    img=cv2.imread(filename)
    #crop image
    img=img[x0:x1,y0:y1]
    #resize image
    img=cv2.resize(img,(size_x,size_y),cv2.INTER_CUBIC)
    if len(save_to_dir)>0:
        dstdir=save_to_dir+'/'+os.path.dirname(filename)
        try:
            os.makedirs(dstdir)
        except:
            pass
        cv2.imwrite(save_to_dir+'/'+filename,img)
    return img
    
        
parser=argparse.ArgumentParser('Trains a dataset in the route')
parser.add_argument('route',help=' Dataset directory ',type=str)
parser.add_argument('cropped',help=' Dataset Cropped directory',default='cropped',type=str)
args=parser.parse_args()
route=args.route
cropped_route=args.cropped
batch_size=50
tamx=32
tamy=32
seed=0

#read all CSVs
data=readAllCSVs(route)
print (data) 
print (len(data))

#read,crop and save all images
c=0
data_cropped=[]
for dat in data:
    load_and_preprocess(dat[0],dat[3],dat[4],dat[5],dat[6],tamx,tamy,cropped_route)
    #c+=1
    #cropped=['cropped/'+dat[0],dat[7]]
    #sp=dat[0].split('/')
    #cropped=[ sp[len(sp)-1],dat[7]]
    #data_cropped.append(cropped)

