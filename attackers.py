import numpy as np
import tensorflow as tf

class FGSMAttacker():
    '''
        A class for implementing the Fast Gradient Sign Method attack (Goodfellow et al, 2014) on a given model 
    '''
    def __init__(self):
        pass

    @tf.function
    def get_adversarial_images(self, images,labels, model, loss_function, epsilon,num_iters):
        '''
            Given an input image, perturb it in accordance with FGSM
            ======
            Args:
            images (np.ndarray): Numpy array of shape 
            n_images x height x width x channels
            labels (np.ndarray): Numpy array of shape
            n_images x 1
            model: A function that maps an image (shape mentioned above) to a label of shape n_classes x 1
            loss_function: A function that returns the loss given labels and predictions
            epsilon: Magnitude of perturbation
            num_iters: Iteratively applies the FGSM perturbation num_iters times
        '''

        delta_total = np.zeros(np.shape(images))
        for i in range(num_iters):
            y_pred = model(images)
            loss = loss_function(labels, y_pred)
            delta = tf.gradients(loss, [images])[0]
            delta = epsilon * tf.sign(delta)
            delta_total += delta
            images = images + delta
            images = tf.clip_by_value(images, clip_value_min=0, clip_value_max=255)
        return images, delta_total