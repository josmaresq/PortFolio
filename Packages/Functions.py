# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:07:28 2023

@author: José Maanuel Marrón Esquivel
"""

import numpy as np
import logging
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import models


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
    CNN.add(layers.Dense(256))
    CNN.add(layers.BatchNormalization())
            
    logging.info('Capa densa con Dropout 0.3 y tamaño de: 256')
    
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