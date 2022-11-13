import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import requests
from tqdm import tqdm
import numpy as np
import numpy as np
from keras.layers import DepthwiseConv2D
from keras.losses import BinaryCrossentropy
from PIL import Image
from attackers import FGSMAttacker
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
            temp_ind_losses.append(BinaryCrossentropy(labels, preds).numpy())
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
    



class Ensemble():
    '''
        Creates an ensemble of models included in the 'models' list
        Confidence scores are averaged
    '''
    def __init__(self, models):
        ''' Parameters
        ----------
        models : list
            List of models to be included in the ensemble
            Each model should be of type keras.models.Model
        '''
        self.models = models
        self.len_models = len(models)
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
        combine_methods = {2: self._combine_mean, 3: self._combine_mean_excluding_outliers} # if there are 2 models, use mean, if there are 3 models, use mean excluding outliers
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


def evaluate_adversarial(model_for_generation,models_for_prediction, image_paths, labels):
    correct_count = np.zeros(len(models_for_prediction))
    avg_loss = np.zeros(len(models_for_prediction))
    attacker = FGSMAttacker()
    predictions = {}
    for image_path, label in tqdm(zip(image_paths, labels)):
        image = Image.open(image_path).resize((100,100))
        image = np.expand_dims(np.array(image), axis=0)

        image = tf.convert_to_tensor(image)
        image = tf.cast(image, tf.float64)
        label = tf.convert_to_tensor([label])
        image, _ = attacker.get_adversarial_images(image, label, model_for_generation, tf.keras.losses.binary_crossentropy, 0.1, num_iters=30)
        data_info = {}
        gt = np.argmax(label)
        data_info['gt'] = gt
        for i, model in enumerate(models_for_prediction):
            conf = model.predict(image)[0]
            data_info[model.name1] = conf
            loss = np.log(conf[0]) if gt == 0 else np.log(conf[1])
            avg_loss[i] += loss
            pred = conf[0] <= 0.5
            if pred == gt:
                correct_count[i] += 1
        predictions[image_path] = data_info
    acc = 100*correct_count/len(image_paths)
    avg_loss = -avg_loss/len(image_paths)
#     for i, model in enumerate(models_for_prediction):
#         print('Generated on:', model_for_generation.name1, 'Evaluated on:', model.name1, 'Loss:', avg_loss[i], 'Acc:', acc[i], '%')
    return avg_loss, acc, predictions


def get_ensemble_predictions(predictions):
    generate_on_model_names = predictions.keys()
    some_model_name = list(predictions.keys())[0] 
    some_image_path = list(predictions[some_model_name].keys())[12]
    constituent_model_names = set(predictions[some_model_name][some_image_path].keys()) - set(['gt'])
    constituent_model_names = list(constituent_model_names)
    print(constituent_model_names)
    for model_name in generate_on_model_names:
        for image_path in predictions[model_name].keys():
            preds = []
            for constituent_model_name in constituent_model_names:
                preds.append(predictions[model_name][image_path][constituent_model_name])
            pred_e_mean = np.mean(preds, axis=0)
            pred_e_mean_outlier = np.mean(drop_outliers(preds), axis=0)
            predictions[model_name][image_path]['3LP + VGG16'] = np.mean([preds[0], preds[1]], axis=0)
            predictions[model_name][image_path]['Vgg16 + ResNet50'] = np.mean([preds[2], preds[0]], axis=0)
            predictions[model_name][image_path]['ResNet50 + 3LP'] = np.mean([preds[1], preds[2]], axis=0)
            predictions[model_name][image_path]['Ensemble1'] = pred_e_mean
            predictions[model_name][image_path]['Ensemble2'] = pred_e_mean_outlier

    return predictions