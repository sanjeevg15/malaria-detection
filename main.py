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
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, Input, MaxPool2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import np_utils
from matplotlib import pyplot as plt
from models import *
from PIL import Image
from tqdm import tqdm

from helpers import *

root_dir = '/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images'
uninfected_folder = os.path.join(root_dir, 'Uninfected')
infected_folder = os.path.join(root_dir, 'Parasitized')
training_data = image_dataset_from_directory(root_dir, batch_size = 32, label_mode='categorical', image_size=(100,100), validation_split=0.2, subset="training", seed=0)
validation_data = image_dataset_from_directory(root_dir, batch_size= 32, label_mode='categorical', image_size=(100,100), validation_split=0.2, subset="validation", seed=0)

weight_files = ['../trained_models/best_model.hdf5','../trained_models/best_model_3p.hdf5','../trained_models/best_model_v.hdf5', '../trained_models/best_model_r.hdf5']

opt = Adam(learning_rate=1e-3, beta_1=0.99, beta_2=0.999)
models = [model_3l, model_3lp, model_v, model_r]
weight_files = ['../input/malariamodel/best_model.hdf5','../input/malariamodel/best_model_3p.hdf5','../input/malariamodel/best_model_v.hdf5', '../input/malariamodel/best_model_r.hdf5']
for model, weight_file in zip(models, weight_files):
    model(np.zeros([1,100,100,3]))
    model.load_weights(weight_file)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#     model.evaluate(validation_data)







class Ensemble():
    '''
        Creates an ensemble of models included in the 'models' list
        Confidence scores are averaged
    '''
    def __init__(self, models):
        self.models = models
        self.len_models = len(models)

    def load_predictions(self, predictions_file_path):
        '''
            Loads and reuses previously predicted values, if available
        '''
        predictions = np.load(predictions_file_path)
        predictions = predictions.item()
        self.predictions = predictions
        return predictions
        
    def predict(self, images, verbose=False):
        '''
            combine ('mean', 'mean_excluding_outliers', 'mode')
        '''

        predictions = []
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        elif len(images.shape) == 4:
            pass
        else:
            print("Input should be either of rank 3 or rank 4, but got" , " ", len(images.shape))
    
        for model in self.models:
            prediction = model.predict(images)
            predictions.append(prediction[0])
#             print(model.name1, ' ', prediction)
        combine_methods = {2: self._combine_mean, 3: self._combine_mean_excluding_outliers}
        combine_method = combine_methods[self.len_models]
        predictions = np.array(predictions)
        prediction = combine_method(predictions)
        if verbose:
            print(predictions)
        return prediction

    def _combine_mean(self, predictions):
        prediction = np.mean(predictions, axis=0)
        return [prediction]
 
    def _combine_mean_excluding_outliers(self, predictions):
        confs_0 = predictions[:,0]
        c1, c2 = self._cluster_1d(confs_0)
        c = c1 if len(c1) > len(c2) else c2
        conf = np.mean(c)
        prediction = np.array([conf, 1 - conf])
        return [prediction]
    
    @staticmethod
    def _cluster_1d(X):
        '''
            X (numpy array): Data to be clustered
        '''
        X = np.sort(X)
        diffs = np.array([ X[i+1] - X[i] for i in range(len(X) - 1)])
        index = np.argmax(diffs)
        c1 = X[:index + 1]
        c2 = X[index + 1:]
        return c1, c2


    def eval(self, dataset, batch_size=32, intermediate_function=None):
        for images, labels in dataset:
            images_adv, _ = intermediate_function(images, labels)
        for image,label in zip(images_adv, labels):
            prediction = self.predict(image)
            
    def evaluate(self, images, labels, intermediate_function=None, verbose=False):
        '''
            images (array of rank 3 or 4 or list of file paths)
            labels (array of rank 1 or 2 or list of class names)
        '''

        
        if type(images[0]) == str:
            loss, acc = self.evaluate_file_names(images, labels, intermediate_function=intermediate_function, verbose=verbose)
        else:
            loss, acc = self.evaluate_array(images, labels, intermediate_function=intermediate_function, verbose=verbose)
        
        return loss, acc
        
    def evaluate_file_names(self, images, labels, intermediate_function=None, verbose=False):
            
        
        correct_count = 0
        avg_loss = 0
        for image_name, label in tqdm(zip(images, labels)):
            image = Image.open(image_name).resize((100,100))
            image = np.expand_dims(np.array(image), axis=0)
            image = tf.convert_to_tensor(image)
            print('l1 ', label)
            label = [label]
            print('l2', label)
            print(label)
            if intermediate_function:
                image = intermediate_function(image, label)
            
            conf = self.predict(image)
            gt = np.argmax(label)
            loss = np.log(conf[0]) if gt == 0 else np.log(conf[1])
            if verbose:
                print('Parasitized Conf: ', conf, '   GT: ', gt,'   Loss: ', loss)
            avg_loss += loss
            pred = conf[0] <= 0.5
            if pred == gt:
                correct_count += 1
        acc = 100*correct_count/len(images)
        avg_loss = -avg_loss/len(images)
        return avg_loss, acc
    
    def evaluate_array(self, images, labels, intermediate_function=None, verbose=False):
        correct_count = 0
        avg_loss = 0
        for image, label in tqdm(zip(images, labels)):
            image = tf.convert_to_tensor(image)
            label = [label]
            if intermediate_function:
                image, _ = intermediate_function(image, label)
                        
            conf = self.predict(np.expand_dims(image,axis=0))[0]
            gt = np.argmax(label)
            loss = gt*np.log(conf) + (1-gt)*np.log(1-conf) 
            avg_loss += loss
            if verbose:
                print('Parasitized Conf: ', conf, '   GT: ', gt,'   Loss: ' , loss)
           
            pred = conf <= 0.5
            if pred == gt:
                correct_count += 1
        avg_loss = -avg_loss/images.shape[0]
        accuracy = 100*correct_count/images.shape[0]
        return avg_loss, accuracy
