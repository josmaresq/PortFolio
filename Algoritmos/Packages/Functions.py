# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:07:28 2023

@author: José Maanuel Marrón Esquivel
"""

import numpy as np
import logging
from tensorflow.keras import layers
from tensorflow.keras import models
import os

def create_dir(directory):
	if not os.path.isdir(directory):
		try:
			os.mkdir(directory)
		except OSError:
			print ("Creation of the directory %s failed" % directory)
		else:
			print ("Successfully created the directory %s " % directory)


def class_weights_lab(labels):
    unique,counts = np.unique(labels, return_counts=True)
    i=np.argmax(counts)
    c_weights = {}
    
    for j in range(len(unique)):
        c_weights[j] = counts[i]/counts[j]
    print(c_weights)
    return(c_weights)

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

def plot_confusion_matrix(cm,
                          target_names,
                          dir_image,
                          title='Predictions',
                          cmap=None,
                          normalize=True,
                         metric = None,
                         ):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(1800.0/float(DPI),1800.0/float(DPI))
    
    dpi = 600
    fontsize = 30
    
    kappa = metric #0.6613

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
        
    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title, fontsize=25)
    #plt.colorbar()
    
    cb = plt.colorbar()
    for t in cb.ax.get_yticklabels():
         t.set_fontsize(fontsize)
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45,fontsize=fontsize)
        plt.yticks(tick_marks, target_names,fontsize=fontsize)

    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=fontsize)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize=fontsize)

    plt.tight_layout()
    plt.ylabel('Pathologist predictions',fontsize=fontsize)
    plt.xlabel('Model predictions $\kappa$={:0.4f}'.format(kappa),fontsize=fontsize)
    
    plt.tight_layout()
    plt.savefig(dir_image, format='svg',dpi=dpi)


