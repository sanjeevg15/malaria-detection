from tensorflow.python.training.tracking import base
import numpy as np
from PIL import Image
import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D

kernel_weights = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
kernel_weights = np.expand_dims(kernel_weights, axis=-1)
kernel_weights = np.repeat(kernel_weights, 3, axis=-1) # apply the same filter on all the input channels
kernel_weights = np.expand_dims(kernel_weights, axis=-1)  # for shape compatibility reasons

gaussian_blur = DepthwiseConv2D(3, use_bias=False, weights=[kernel_weights], padding='same')

def crop_center(img,cropx,cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def get_snr_malarial_img(img):
    img_non_zero = crop_center(img, 70, 70)
    mu = np.mean(img_non_zero)
    sigma = np.std(img_non_zero)
    snr = mu/sigma
    snr = np.around(snr, decimals=4)
    return snr

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

def generate_fake_image(size=[100,100,3], class_name='Parasitized', base_img=None, blob_radius=0.1, blob_color='auto', blob_center='img_center'):
    '''
        Generate a malarial cell like image to try and fool the CNN
        ===========================================================
        Arguments:
        size (list like): Desired size of the generated images 
        class_name (String): 'Parasitized' or 'Uninfected'
        base_img (np.ndarray): Image to be used as base image
        Size of this image overrides the 'size' argument
        ===========================================================
        Returns:
        img (numpy.ndarray): Fake image of the desired class
    '''
    if not base_img:
        base_img = Image.open('C:/Users/sanje/Documents/Datasets/Malaria/cell_images/cell_images/Uninfected/C1_thinF_IMG_20150604_104722_cell_9.png')
        base_img = base_img.resize((100,100))
    r,g,b = base_img.split()
    # avg_r = np.sum(np.array(r))/np.sum(np.array(r)!=0)
    # avg_g = np.sum(np.array(g))/np.sum(np.array(g)!=0)
    # avg_b = np.sum(np.array(b))/np.sum(np.array(b)!=0)

    # rgb = np.array([avg_r, avg_g, avg_b])

    base_img = np.array(base_img)

    blob_radius = blob_radius*np.min(np.array(base_img).shape[0])
    while(True):
        blob_center = get_random_location([base_img.shape[0], base_img.shape[1]])
        if np.any(base_img[blob_center[0], blob_center[1]] != np.zeros(3)):
            break

    img_out = add_blob(base_img, blob_center, blob_radius)
    return img_out

def get_random_location(dims):
    location = []
    for dim in dims:
        loc = int(np.random.rand()*dim)
        location.append(loc)

    return location


def add_blob(img, blob_center='img_center', blob_radius=8, rgb=[150,60,150]):
    img_out = np.copy(img)
    r, g, b = rgb
    if blob_center=='img_center':
        blob_center = img_out.shape//2
    for i in range(img_out.shape[0]):
        for j in range(img_out.shape[1]):
            d = np.linalg.norm(np.array([i,j]) - np.array(blob_center))
            if d <= blob_radius:
                if np.any(img_out[i][j][:] != np.array([0,0,0])):
                    img_out[i][j][:] = np.array([r,g,b])

    return img_out


def get_centroid(img):
    '''
        img(np.ndarray or PIL image object): Image
    '''
    img = np.array(img)
    n = 0
    img = img[:,:,0]
    centroid = np.zeros(2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > 0:
                n = n + 1
                centroid = (n*centroid + np.array([i,j]))/(n+1)
    return centroid