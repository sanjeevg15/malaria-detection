import copy
import os
import pickle
import shutil

from sklearn.cluster import KMeans

import cv2
import seaborn as sn
import tensorflow as tf
from attackers import *
from helpers import *
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

root_dir = '/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images'
uninfected_folder = os.path.join(root_dir, 'Uninfected')
infected_folder = os.path.join(root_dir, 'Parasitized')
training_data = image_dataset_from_directory(root_dir, batch_size = 32, label_mode='categorical', image_size=(100,100), validation_split=0.2, subset="training", seed=0)
validation_data = image_dataset_from_directory(root_dir, batch_size= 32, label_mode='categorical', image_size=(100,100), validation_split=0.2, subset="validation", seed=0)

weight_files = ['../input/malariamodel/best_model.hdf5','../input/malariamodel/best_model_3p.hdf5','../input/malariamodel/best_model_v.hdf5', '../input/malariamodel/best_model_r.hdf5']

