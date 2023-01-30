# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:22:35 2022

@author: José Manuel Marrón Esquivel
"""
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import EarlyStopping
import argparse

import Packages.Functions as func

import logging


#gpu training config
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

history = History()

argv = sys.argv[1:]

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-d', '--OUTPUT_PATH', help='models output directory',type=str)
parser.add_argument('-pd', '--PATCHES_DIRECTORY', help='patches directory',type=str)
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=22)
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=64)
parser.add_argument('-e', '--EPOCHS', help='number of epochs',type=int, default=50)
parser.add_argument('-mxlr', '--MAX_LR', help='max Learning Rate',type=int, default=0.001)
parser.add_argument('-mnlr', '--MIN_LR', help='min Learning Rate',type=int, default=0.00001)
parser.add_argument('-p', '--PATCH_SIZE', help='patch size',type=int, default=224)


args = parser.parse_args()

total_EXP = 10

# OUTPUT_PATH = 'C:\\Users\\EQUIPO\\Documents\\Adenocarcinoma\\CNN\\iterations_PANDA'


if not os.path.exists(args.OUTPUT_PATH):
    os.mkdir(args.OUTPUT_PATH)

dirname = os.listdir(args.PATCHES_DIRECTORY)


droplist = []

PANDA_patches_dataframe_aux = pd.read_csv(args.PATCHES_DIRECTORY + "\\" + "PANDA_patches_strong_labels.csv", header = None)
PANDA_patches_dataframe = pd.read_csv(args.PATCHES_DIRECTORY + "\\" + "PANDA_patches_strong_labels.csv", header = None)

for i in range(len(PANDA_patches_dataframe_aux)):
    if PANDA_patches_dataframe_aux[1][i] == 0: # Se guardan los índices de las anotaciones a 0
          
          droplist.append(i)
          
PANDA_patches_dataframe.drop(droplist,inplace=True) # Se eliminan los datos con anotacion 0 del dataf

PANDA_patches_dataframe_shuffled = PANDA_patches_dataframe.sample(frac = 1, random_state=(1)).reset_index(drop=True)

long = round((3*len(PANDA_patches_dataframe_shuffled))/4)

PANDA_patches_train_dataframe = PANDA_patches_dataframe_shuffled.iloc[:long][:]
PANDA_patches_valid_dataframe = PANDA_patches_dataframe_shuffled.iloc[long:][:].reset_index(drop=True)
     
print('PANDA train patches found: ', np.all([os.path.isfile(i) for i in PANDA_patches_train_dataframe[0]]))
print('PANDA valid patches found: ',np.all([os.path.isfile(i) for i in PANDA_patches_valid_dataframe[0]]))

# Incremento del dataset

class_weight = func.class_weights_lab(PANDA_patches_train_dataframe[1])
    
train_data_generator = ImageDataGenerator(rescale=1./255, preprocessing_function=func.preprocess_train) #rotation_range=360, fill_mode='reflect')
val_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)
    
    
PANDA_patches_train_dataframe.columns = ['filename', 'class']
PANDA_patches_train_dataframe['class'] = [str(x) for x in PANDA_patches_train_dataframe['class']]
PANDA_patches_valid_dataframe.columns = ['filename', 'class']
PANDA_patches_valid_dataframe['class'] = [str(x) for x in PANDA_patches_valid_dataframe['class']]
    
train_generator = train_data_generator.flow_from_dataframe(PANDA_patches_train_dataframe, x_col='filename', y_col='class', class_mode="categorical", target_size=(args.PATCH_SIZE,args.PATCH_SIZE), color_mode='rgb', batch_size=args.BATCH_SIZE, shuffle=True)
validation_generator = val_data_generator.flow_from_dataframe(PANDA_patches_valid_dataframe, x_col='filename', y_col='class', class_mode="categorical", target_size=(args.PATCH_SIZE,args.PATCH_SIZE), color_mode='rgb', batch_size=args.BATCH_SIZE, shuffle=False) #, save_to_dir='D:\\Repositorios\\ProstateCancer\\CNN\\Prueba')
    
IMAGE_SIZE = (224, 224)
IMG_SHAPE = IMAGE_SIZE + (3, )  

list_LR = [args.MAX_LR]

n = args.MAX_LR

while n > args.MIN_LR:
    n = n**-1
    list_LR.append(n)

for it in range(total_EXP):
    
    total_PATH = args.OUTPUT_PATH + '\\' + 'IT_' + str(it)
    
    if not os.path.exists(total_PATH):
        os.mkdir(total_PATH)
        
    logging.basicConfig(filename= total_PATH + '\\registro.log', level=logging.INFO)

    for lr in list_LR:
        
        total_PATH = total_PATH + '\\' + 'LR_' + str(lr)
        
        if not os.path.exists(total_PATH):
            os.mkdir(total_PATH)
        
        CNN = func.create_network(it)
    
        for l in CNN.layers:
            print(l.name, l.trainable)
    
        checkpoint = ModelCheckpoint(total_PATH + '\\model-{epoch:03d}-{val_loss:03f}-{val_accuracy:03f}.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto') #, mode='min'
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        
        callbacks_list = [checkpoint, history, es] #, reduce_lr] #, es] #, reduce_lr]
    
        opt = tf.keras.optimizers.Adam(learning_rate=lr)    
    
        CNN.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
        # CNN.summary()
    
        history = CNN.fit(
                    train_generator,
                    steps_per_epoch= train_generator.n/args.BATCH_SIZE,
                    epochs=args.EPOCHS,
                    validation_data=validation_generator,
                    validation_steps= validation_generator.n/args.BATCH_SIZE,
                    callbacks=callbacks_list,
                    class_weight=class_weight
                    ,        workers = 12
                    )
        acc = history.history['accuracy']
    
        val_acc = history.history['val_accuracy']
    
        loss = history.history['loss']
        val_loss = history.history['val_loss']