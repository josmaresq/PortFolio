# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:09:33 2023

@author: José Manuel Marrón Esquivel
"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import statistics as st
import os
import numpy as np
import sys
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import load_model
import argparse


gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)

history = History()

argv = sys.argv[1:]

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
# parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=135)
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=8)

args = parser.parse_args()

# N_EXP = args.N_EXP
BATCH_SIZE = args.BATCH_SIZE


