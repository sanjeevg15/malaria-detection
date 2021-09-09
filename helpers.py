# CIFAR - 10

import pickle
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
import pandas as pd
import requests
from tqdm import tqdm
import numpy as np
import numpy as np
from keras.layers import DepthwiseConv2D
import tensorflow as tf

kernel_weights = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
kernel_weights = np.expand_dims(kernel_weights, axis=-1)
kernel_weights = np.repeat(kernel_weights, 3, axis=-1) # apply the same filter on all the input channels
kernel_weights = np.expand_dims(kernel_weights, axis=-1)  # for shape compatibility reasons

gaussian_blur = DepthwiseConv2D(3, use_bias=False, weights=[kernel_weights], padding='same')

def add_noise(noise_type,image,params=[0,0.1]):

    '''Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gaussian'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    '''

    image2 = np.copy(image)
    if np.max(image2) > 1:
        image2 = image2/255
    
    
    if noise_type == "gaussian":
        row,col,ch= image2.shape
        mean = params[0]
        var = params[1]
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy_img = image2 + gauss
        noisy_img = 255*noisy_img
        noisy_img = noisy_img.astype("uint8")
        return noisy_img
    
    elif noise_type == "s&p":
        row,col,ch = image2.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image2)
        # Salt mode
        num_salt = np.ceil(amount * image2.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image2.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image2.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image2.shape]
        out[coords] = 0
        return out
    
    elif noise_type == "poisson":
        vals = len(np.unique(image2))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_img = np.random.poisson(image2 * vals) / float(vals)
        noisy_img = 255*noisy_img
        noisy_img = noisy_img.astype("uint8")
        return noisy_img
    
    elif noise_type =="speckle":
        mean = params[0]
        var = params[1]
        row,col,ch = image2.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)   
        gauss = gauss*var + mean
        noisy_img = image2 + image2*gauss
        noisy_img = 255*noisy_img
        noisy_img = noisy_img.astype("uint8")
        return noisy_img

def get_acc_loss(predictions):
    generate_on_models = list(predictions.keys())
    evaluate_on_models = list(predictions[list(predictions.keys())[0]][list(predictions[generate_on_models[0]].keys())[0]].keys()) 
    evaluate_on_models.remove('gt')
    print(evaluate_on_models)
    mean_loss = {}
    median_loss = {}
    acc = {}
    for model1_name in generate_on_models:
        labels = [[0.0, 1.0] if predictions[model1_name][image_path]['gt'] == 1 else [1.0, 0.0] for image_path in predictions[model1_name].keys()]
        temp_ind_losses = []
        temp_acc = []
        
        for model2_name in evaluate_on_models:
            preds = [predictions[model1_name][image_path][model2_name] for image_path in predictions[model1_name].keys()]
            temp_ind_losses.append(tf.losses.binary_crossentropy(labels, preds).numpy())
            temp_acc.append(get_accuracy(labels, preds))
        mean_loss[model1_name] = [np.mean(i) for i in temp_ind_losses]
        median_loss[model1_name] = [np.median(i) for i in temp_ind_losses]
        acc[model1_name] = temp_acc
    return acc, mean_loss, median_loss

def get_accuracy(labels, predictions):
    count = 0
    for prediction, label in zip(predictions, labels):
        if np.argmax(prediction) == np.argmax(label):
            count += 1
    return 100*count/len(predictions)