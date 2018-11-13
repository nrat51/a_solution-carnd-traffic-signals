import os
import sys
import argparse	
from keras.preprocessing.image import DirectoryIterator,ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,MaxPooling2D,Convolution2D,Flatten,BatchNormalization
from keras.callbacks import TensorBoard,ModelCheckpoint	
import numpy as np
import random

#parseado de linea de comandos
parser=argparse.ArgumentParser()
parser.add_argument("dataset",help="Directory with Dataset",type=str)
parser.add_argument("execution_name",help="Name for the execution",type=str)
parser.add_argument("batch_size",help="Batch size (default=10)",type=int,nargs='?',default=10)
parser.add_argument("epochs",help=" Number of epochs (default=10)",type=int,nargs='?',default=10)
parser.add_argument("image_dim",help="(dimXdim) image input size (default=32) --> 32x32",type=int,nargs='?',default=32)
parser.add_argument("--data_augment",help="Use data augmentation ",action='store_true',default=False)
args=parser.parse_args()

#parametros iniciales
route=args.dataset
execution_name=args.execution_name
batch_size=args.batch_size
epochs=args.epochs
data_augmentation=args.data_augment
tam=args.image_dim
img_size=(tam,tam)
input_shape=(img_size[0],img_size[1],3)
optimizer=Adam(lr=0.0002) #lr=0.0001)
print (args)

#todas las seed (random,numpy, myseed)
myseed=0
np.random.seed(myseed)
random.seed(myseed)


#generador datos
if data_augmentation:
    image_generator=ImageDataGenerator(
    #zca_whitening=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
    horizontal_flip=False, #sino confundiría giros izquierda derecha
    vertical_flip=False, # en señales no debemos encontrarlas al reves
    validation_split=0.2)
else:
    image_generator=ImageDataGenerator(validation_split=0.2)

#iterador de directorio para train
train_directory_generator=DirectoryIterator(directory=route,
                                    image_data_generator=image_generator,
                                    target_size=img_size,
                                    class_mode='categorical',
                                    batch_size=batch_size,
                                    interpolation='bicubic',
                                    color_mode='rgb',
                                    dtype=float,
                                    shuffle=True,
                                    seed=myseed,
                                    data_format='channels_last',
                                    subset='training')
nitems_train=len(train_directory_generator.classes)

#iterador de directorio para validacion
val_directory_generator=DirectoryIterator(directory=route,
                                    image_data_generator=image_generator,
                                    target_size=img_size,
                                    class_mode='categorical',
                                    batch_size=batch_size,
                                    interpolation='bicubic',
                                    color_mode='rgb',
                                    dtype=float,
                                    shuffle=True,
                                    seed=myseed,
                                    data_format='channels_last',
                                    subset='validation')
nitems_val=len(val_directory_generator.classes)                                    

#comprobamos todas las clases diferentes que ha encontrado el generador
distinct_classes=[]
for i in train_directory_generator.classes:
    if not str(i) in distinct_classes:
             distinct_classes.append(str(i))
for i in range(0,len(distinct_classes)):
    distinct_classes[i]=str("%05d"%int(distinct_classes[i]))
print(distinct_classes)                                 
num_classes=len(distinct_classes)


#modelo
def model1(input_shape,num_classes):
    model=Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(32,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu',name='conv0'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid',name='pool0'))
    model.add(Flatten())
    model.add(Dense(num_classes,activation='softmax'))
    return model

def model2(input_shape,num_classes):
    model=Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(256,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',name='conv0'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid',name='pool0'))
    model.add(BatchNormalization())
    model.add(Convolution2D(128,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',name='conv1'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid',name='pool1'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu',name='conv2'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='valid',name='pool2'))
    model.add(Flatten())
    model.add(Dense(num_classes,activation='softmax'))
    return model
    
    
model=model2(input_shape,num_classes)    
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy','mse'])
model.summary()

#callbacks, checkpoint y Tensorboard
tsb=TensorBoard(log_dir='TSB/'+execution_name,write_graph=True,write_images=True,update_freq='batch') #,histogram_freq=1,embeddings_freq=1)
check=ModelCheckpoint('saved/'+execution_name+'_{epoch}_acc-{val_acc}_loss-{val_loss}.model',save_best_only=False)
mycallbacks=[tsb,check]

model.fit_generator(generator=train_directory_generator
                   ,steps_per_epoch=int(nitems_train/batch_size)
                   ,epochs=int(epochs)
                   ,callbacks=mycallbacks
                   ,validation_data=val_directory_generator
                   ,validation_steps=int(nitems_val/batch_size)
                   ,verbose=1
                   )
                   
                   
                   