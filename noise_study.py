import matplotlib.pyplot as plt
from helpers import add_noise
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from models import *

# parser = ArgumentParser(description='Noise study')
# parser.add_argument('model_name', metavar='model', type=str, nargs=1)

def noise_study(model, data, class_index=1, n_images=10, step_size=0.01,num_steps=20, class_names=['P','U'],data_root='./MalariaDataset'):
    '''
        Adds noise gradually to the images in the dataset and measures average model confidence for the true class in the dataset

        ====== ARGS ======
        model() : Model whose performance you want to check for robustness to noise
        data(tuple): (X,y) where X is an np.ndarray of size mxn and y is the array containing labels for images in X
        class_index (int): Indicates which class to sample images from
        n_images (int) : No. of images to sample
        step_size (float): Increase standard deviation by this amount every step
        num_steps (int): No. of steps of standard deviation increment
        class_names (list): List of class names
        data_root (string): Root folder for the dataset  

        ====== RETURNS ======
        graphs (list of np.ndarray): Array containing the model's confidence that the image belongs to class specified by class index for all values of standard deviation
    '''

    graphs = []
    k = 0
    X,y = data
    for image, label in tqdm(zip(X,y)):
        conf_noisy = []
        k += 1
        if k > n_images:
            break
        if label == class_index:
            gt = class_names[int(label)][0]
            parasitized_confidence_orig = model.predict(np.expand_dims(image, axis=0))[0][0]
            if parasitized_confidence_orig < 0.5:
                original_pred = 'Uninfected'
            else:
                original_pred = 'Parasitized'

            for i in range(num_steps):
                std = step_size*i
                noisy_img = add_noise('speckle', image, [0,std])
                parasitized_conf_noisy = model.predict(np.expand_dims(noisy_img, axis=0))[class_index][0]
                conf_noisy.append(parasitized_conf_noisy)
            graphs.append(np.array(conf_noisy))

            plt.clf()
            plt.axis('on')
            plt.plot(np.arange(10)*0.01, conf_noisy)
            plt.ylim([0,1.5])
        #     plt.title('Parasitized Confidence Level v/s Noise Level')
            plt.ylabel('Model Confidence for Class: ' + class_names[class_index])
            plt.xlabel('Standard Deviation of Noise Added ($\sigma$)')
            save_path = ('./conf_plot_' + original_pred+'.png')
            plt.savefig(save_path, bbox_inches='tight', facecolor='w')
    return graphs

# if __name__=='__main__':
#     model = model_3lp
#     data = 
#     noise_study(model, data)