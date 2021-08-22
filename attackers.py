import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import tensorflow as tf

class FGSMAttacker():
    def __init__(self):
        pass

    @tf.function
    def get_adversarial_images(self, images,labels, model, loss_function, epsilon,num_iters):
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