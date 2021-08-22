# %% [code]
import numpy as np
import numpy as np
from keras.layers import DepthwiseConv2D

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