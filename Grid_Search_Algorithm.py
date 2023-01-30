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
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import History

import logging

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

import argparse
argv = sys.argv[1:]

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=22)
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=64)
parser.add_argument('-e', '--EPOCHS', help='number of epochs',type=int, default=150)

args = parser.parse_args()

N_EXP = args.N_EXP

total_EXP = 10

class settings:
    OUTPUT_PATH = 'C:\\Users\\EQUIPO\\Documents\\Adenocarcinoma\\CNN\\iterations_PANDA'
    BATCH_SIZE = 16
    EPOCHS = 40
    DATA_AUGMENTATION = True
    #PREPROCESSING = False
    PATCH_SIZE = 224
    LR=0.001

settings = settings()

if not os.path.exists(settings.OUTPUT_PATH):
    os.mkdir(settings.OUTPUT_PATH)

original_dir = "C:\\Users\\EQUIPO\\Documents\\Adenocarcinoma\\PANDA\\patches_annotated"

dirname = os.listdir(original_dir)


droplist = []

PANDA_patches_dataframe_aux = pd.read_csv(original_dir + "\\" + "PANDA_patches_strong_labels.csv", header = None)
PANDA_patches_dataframe = pd.read_csv(original_dir + "\\" + "PANDA_patches_strong_labels.csv", header = None)

for i in range(len(PANDA_patches_dataframe_aux)):
    if PANDA_patches_dataframe_aux[1][i] == 0: # Se guardan los índices de las anotaciones a 0
          
          droplist.append(i)
          
PANDA_patches_dataframe.drop(droplist,inplace=True) # Se eliminan los datos con anotacion 0 del dataf

PANDA_patches_dataframe_shuffled = PANDA_patches_dataframe.sample(frac = 1, random_state=(1)).reset_index(drop=True)

long = round((3*len(PANDA_patches_dataframe_shuffled))/4)

PANDA_patches_train_dataframe = PANDA_patches_dataframe_shuffled.iloc[:long][:]
PANDA_patches_valid_dataframe = PANDA_patches_dataframe_shuffled.iloc[long:][:].reset_index(drop=True)
    
dir_PATH = settings.OUTPUT_PATH + '\\' + 'N_EXP_' + str(N_EXP)

if not os.path.exists(dir_PATH):
    os.mkdir(dir_PATH)
     
print('PANDA train patches found: ', np.all([os.path.isfile(i) for i in PANDA_patches_train_dataframe[0]]))
print('PANDA valid patches found: ',np.all([os.path.isfile(i) for i in PANDA_patches_valid_dataframe[0]]))


def create_network(iteracion):
    
    logging.info('RED NUMERO ' + str(iteracion))
    
    CNN = models.Sequential()
    CNN.add(layers.Conv2D(16, 5, input_shape=(224, 224, 3), padding='same', activation='relu')) #Tener en cuenta tamaño de la imagen
    CNN.add(layers.BatchNormalization()) #Añadido
    CNN.add(layers.MaxPooling2D(pool_size=2))
    
    logging.info('Capa conv2D 16 filtros, tamaño de kernel: 5')
    
    if iteracion > 0:
        contador = 1
        
        for i in range(iteracion):
            if i <= 1:
                CNN.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
                CNN.add(layers.BatchNormalization()) #Añadido
                CNN.add(layers.MaxPooling2D(pool_size=2))
                
                logging.info('Capa conv2D 32 filtros, tamaño de kernel: 3')
                
            else:
                
                if (i % 2) == 0 and i != 2:
                    
                    contador = contador + 1
                    
                CNN.add(layers.Conv2D(2**(5+contador), 3, padding='same', activation='relu'))
                CNN.add(layers.BatchNormalization()) #Añadido
                CNN.add(layers.MaxPooling2D(pool_size=2, padding = 'same'))
                
                logging.info('Capa conv2D '+ str(2**(5+contador)) + ' filtros, tamaño de kernel: 3')                

                
    CNN.add(layers.Flatten())
    
    logging.info('Capa de Flatten')
    
    CNN.add(layers.Activation('relu'))
    CNN.add(layers.Dropout(0.3))
    CNN.add(layers.Dense(512))
    CNN.add(layers.BatchNormalization())
            
    logging.info('Capa densa con Dropout 0.3 y tamaño de: 512')
    
    CNN.add(layers.Activation('relu'))
    CNN.add(layers.Dropout(0.3))
    CNN.add(layers.Dense(128))
    CNN.add(layers.BatchNormalization()) #Añadido
    
    logging.info('Capa densa con Dropout 0.3 y tamaño de: 128 ')
    
    CNN.add(layers.Activation('relu'))
    CNN.add(layers.Dropout(0.3))
    CNN.add(layers.Dense(64))
    CNN.add(layers.BatchNormalization()) #Añadido
    
    logging.info('Capa densa con Dropout 0.3 y tamaño de: 64 ')
    
    CNN.add(layers.Activation('relu'))
    CNN.add(layers.Dense(3)) #Salida de la CNN, en nuestro caso será 3
    CNN.add(layers.Activation('softmax'))
    
    logging.info('Capa densa de salida con tamaño de 3')

    logging.info(' ')
    
    return CNN

def class_weights_lab(labels):
	unique, counts = np.unique(labels, return_counts=True)
	# elems = dict(zip(unique, counts))
	i = np.argmax(counts)
	c_weights = {}

	for j in range(len(unique)):
		c_weights[j] = counts[i]/counts[j]
	print(c_weights)
	return c_weights

# Incremento del dataset

import albumentations as A
prob = 0.5
pipeline_transform = A.Compose([
	A.VerticalFlip(p=prob),
	A.HorizontalFlip(p=prob),
	A.RandomRotate90(p=prob),
	A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),p=prob)
	])

def preprocess_train(image):

    transformed = pipeline_transform(image=image)
    image = transformed['image']
    
    return image

class_weight = class_weights_lab(PANDA_patches_train_dataframe[1])
    
train_data_generator = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_train) #rotation_range=360, fill_mode='reflect')
val_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)
    
    
PANDA_patches_train_dataframe.columns = ['filename', 'class']
PANDA_patches_train_dataframe['class'] = [str(x) for x in PANDA_patches_train_dataframe['class']]
PANDA_patches_valid_dataframe.columns = ['filename', 'class']
PANDA_patches_valid_dataframe['class'] = [str(x) for x in PANDA_patches_valid_dataframe['class']]
    
train_generator = train_data_generator.flow_from_dataframe(PANDA_patches_train_dataframe, x_col='filename', y_col='class', class_mode="categorical", target_size=(settings.PATCH_SIZE,settings.PATCH_SIZE), color_mode='rgb', batch_size=settings.BATCH_SIZE, shuffle=True)
validation_generator = val_data_generator.flow_from_dataframe(PANDA_patches_valid_dataframe, x_col='filename', y_col='class', class_mode="categorical", target_size=(settings.PATCH_SIZE,settings.PATCH_SIZE), color_mode='rgb', batch_size=settings.BATCH_SIZE, shuffle=False) #, save_to_dir='D:\\Repositorios\\ProstateCancer\\CNN\\Prueba')
    
IMAGE_SIZE = (224, 224)
IMG_SHAPE = IMAGE_SIZE + (3, )  

opt = tf.keras.optimizers.Adam(learning_rate=0.1)    

for it in range(total_EXP):
    
    total_PATH = settings.OUTPUT_PATH + '\\' + 'N_EXP_' + str(N_EXP) + '\\' + 'IT_' + str(it)
    
    if not os.path.exists(total_PATH):
        os.mkdir(total_PATH)
        
    logging.basicConfig(filename= total_PATH + '\\registro.log', level=logging.INFO)
    
    CNN = tf.keras.applications.DenseNet121(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')

    inputs = tf.keras.Input(shape=(224, 224, 3))
    prediction_layer = layers.Dense(3, activation="softmax")
    tf.keras.layers.Activation('softmax')
    x = CNN(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(640)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # x = tf.keras.layers.Dense(3)(x)
    outputs = prediction_layer(x)
    CNN = tf.keras.Model(inputs, outputs)
    
    for l in CNN.layers:
        print(l.name, l.trainable)
     
    
    checkpoint = ModelCheckpoint(settings.OUTPUT_PATH + '\\' + 'N_EXP_' + str(N_EXP) + '\\' + 'IT_' + str(it) + '\\model-{epoch:03d}-{val_loss:03f}-{val_accuracy:03f}.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto') #, mode='min'
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    callbacks_list = [checkpoint, history ] #, reduce_lr] #, es] #, reduce_lr]

    
    CNN.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    # CNN.summary()
    
    history = CNN.fit(
                    train_generator,
                    steps_per_epoch= train_generator.n/settings.BATCH_SIZE,
                    epochs=settings.EPOCHS,
                    validation_data=validation_generator,
                    validation_steps= validation_generator.n/settings.BATCH_SIZE,
                    callbacks=callbacks_list,
                    class_weight=class_weight
                    ,        workers = 12
                    )
    acc = history.history['accuracy']
    
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']