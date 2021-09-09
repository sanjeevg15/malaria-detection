import copy
import os
import pickle
import shutil

from sklearn.cluster import KMeans

import cv2
import seaborn as sn
import tensorflow as tf
from attack_py import *
from attackers import *
from helper import *
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, Input, MaxPool2D)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image_dataset_from_directory
from keras.utils import np_utils
from matplotlib import pyplot as plt
from models import *
from PIL import Image
from tqdm import tqdm

from utils import *
%matplotlib inline

opt = Adam(learning_rate=1e-3, beta_1=0.99, beta_2=0.999)
models = [model_3l, model_3lp, model_v, model_r]

for model, weight_file in zip(models, weight_files):
    model(np.zeros([1,100,100,3]))
    model.load_weights(weight_file)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#     model.evaluate(validation_data)