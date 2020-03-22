import numpy as np
import scipy.optimize as opt
import ipdb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
from model import create_model
from tensorflow.keras import models
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from scipy import stats
import os
import copy
from PIL import Image
import pickle

def get_linreg(img,steps):
    ipdb.set_trace()    
    steps-=1
    img = np.delete(img,img.shape[0]-1,0)
    img = np.delete(img,img.shape[1]-1,1)
    img = np.delete(img,0,0)
    img = np.delete(img,0,1)


    center = steps


    left = img[center,:center+1]
    right = img[center,center:]
    up = img[:center+1,center]
    down = img[center:,center]

    
    ipdb.set_trace()

    step_space = np.linspace(0,steps+1,steps+1)

    directions =[left,up,down,right]

    slopes = []
    for direction in directions:
        slope, _, _, _, _ = stats.linregress(step_space,direction)
        slopes.append(abs(slope))


    return np.mean(slopes)


def twoD_GaussianScaledAmp(xy, xo, yo, sigma, amplitude, offset):
    """Function to fit, returns 2D gaussian function as 1D array"""
    x,y = xy
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude*np.exp( - (((x-xo)**2)/(2*sigma**2) + ((y-yo)**2)/(2*sigma**2)))
    return g.ravel()

def getFWHM_GaussianFitScaledAmp(img):
    """Get FWHM(x,y) of a blob by 2D gaussian fitting
    Parameter:
        img - image as numpy array
    Returns:
        FWHMs in pixels, along x and y axes.
    """
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    #Parameters: xpos, ypos, sigma, amp, baseline
    initial_guess = (img.shape[1]/2,img.shape[0]/2,1,1,0)
    # subtract background and rescale image into [0,1], with floor clipping
    bg = np.percentile(img,5)
    img_scaled = np.clip((img - bg) / (img.max() - bg),0,1)
    try:
        popt, pcov = opt.curve_fit(twoD_GaussianScaledAmp, (x, y),
                                   img_scaled.ravel(), p0=initial_guess,
                                   bounds = ((img.shape[1]*0.4, img.shape[0]*0.4, 1, 0.5, -0.1),
                                         (img.shape[1]*0.6, img.shape[0]*0.6, img.shape[1], 1.5, 0.5)))
    except Exception as e:
        print(e)
        return np.nan
    xcenter, ycenter, sigma, amp, offset = popt[0], popt[1], popt[2], popt[3], popt[4]
    FWHM = np.abs(4*sigma*np.sqrt(-0.5*np.log(0.5)))
    return FWHM

#calling example: img is your image
#FWHM = getFWHM_GaussianFitScaledAmp(img)

def load_model(data_shape, weights_path,model_type, label):
    """
    Load and compile VGG model
    args: weights_path (str) trained weights file path
    returns model (Keras model)
    """
    # # either VGG16(), VGG16_keras or BN_VGG
    # if model_type == 'VGG_16':
    #     model = VGG_16_keras(weights_path,data_shape)

    # if model_type == 'BN_VGG':
    #     model = BN_VGG(weights_path,data_shape)
    # if model_type == 'code_BN_VGG':
    #     model = create_model(type='VGG16_BN',pretrained = False, img_shape = data_shape, n_hidden = [4096,4096,1024], dropout = 0.5,
    #      label = label, arr_channels = [], VGG16_top = False, use_gen = False, dropout_arr = [1,1,0], weight_decay = 0.0005)


    print("Loading weights...")
    pdb.set_trace()
    model = models.load_model(weights_path)

    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics = ["accuracy", "top_k_categorical_accuracy"])
    model.summary()

    return model

def one_hot_encoding(y_train, y_test, classes):

    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    return y_train, y_test


def load_data(data_type,label):
    
    if label == 'fine':

        classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data('fine')
        y_train, y_test = one_hot_encoding(y_train, y_test, classes)

        width, height, channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
        x_train = x_train.reshape((x_train.shape[0], width, height, channels))

        width, height, channels = x_test.shape[1], x_test.shape[2], x_test.shape[3]
        x_test = x_test.reshape((x_test.shape[0], width, height, channels))

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        x_train /= 255.0
        x_test /= 255.0

    else:

        classes = 20
        (x_train, y_train), (x_test, y_test) = cifar100.load_data('coarse')
        y_train, y_test = one_hot_encoding(y_train, y_test, classes)

        width, height, channels = x_train.shape[1], x_train.shape[2], x_train.shape[3]
        x_train = x_train.reshape((x_train.shape[0], width, height, channels))

        width, height, channels = x_test.shape[1], x_test.shape[2], x_test.shape[3]
        x_test = x_test.reshape((x_test.shape[0], width, height, channels))

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        x_train /= 255.0
        x_test /= 255.0


    if data_type == 'act':

        data = x_test
        data_shape = data.shape

        return data, data_shape

    if data_type == 'test':
        return x_test, y_test


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def postprocess(deconv):
    if K.image_data_format == 'channels_first':
        deconv = np.transpose(deconv, (1, 2, 0))
    deconv = deconv - deconv.min()
    deconv *= 1.0 / (deconv.max() + 1e-8)
    deconv = deconv[:, :, ::-1]
    uint8_deconv = (deconv * 255).astype(np.uint8)
    img = Image.fromarray(uint8_deconv, 'RGB')
    return img
"""
                idx = 1


                print(layer_list[idx][1])
                print(final_layer)

                while layer_list[idx][1] != final_layer:

                    #pdb.set_trace()
                    deconv_layers[0][0].up(data[sample])
                    deconv_layers[idx][0].up(deconv_layers[idx -1][0].up_data)
                    idx += 1

                    #pdb.set_trace()

                print(elem)
"""

            #for i in range(1, len(deconv_layers)):
            #    deconv_layers[i].up(deconv_layers[i - 1].up_data)


def get_values(img_dict):

    for elem in img_dict:

        for e in img_dict[elem]:
            if e[2] == 0 or e[2] == 64:

                print(e[2])

def deconvolution_loop(deconv_save_path,data):

    while True:


        deconv = pickle.load(open(deconv_save_path,'rb'))
        print(deconv.keys())
        layer_name = input("Insert layer_name: ")
        #layer_name = 'block1_conv2'
        print ('There are {} units in layer {}'.format(len(deconv[layer_name]), layer_name))
        neuron_num = input("Insert unit number: ")
        #neuron_num = 42
        print('layer_name: {}, neuron: #{}'.format(layer_name, neuron_num))

        neuron = deconv[layer_name][int(neuron_num)]

        deconv_img = []
        img_list = []
        overlay_images = []
        deprocess_images_mode = []

        plt.figure(figsize=(10 , 10))

        for idx in range(5):
            deconv_img.append(deprocess_image(neuron[idx][1]))
            img_list.append(neuron[idx][0])
            plt.subplot(3,5,idx+1)
            plt.imshow(deprocess_image(neuron[idx][1]))


        print("läuft")
        for idx in range(5):

            plt.subplot(3,5,idx + 6)
            plt.imshow(data[neuron[idx][0][0]])


        for idx in range(5):

            deprocess_img = deprocess_image(neuron[idx][1])
            overlay_img = copy.deepcopy(data[neuron[idx][0][0]])

            for i in range(deprocess_img.shape[0]):
                for jd in range(deprocess_img.shape[1]):
                    for cd in range(deprocess_img.shape[2]):
                        #print(stats.mode(deprocess_img, axis=None)[0])
                        if deprocess_img[i][jd][cd] <= 119 or deprocess_img[i][jd][cd] >= 128:
                            overlay_img[i][jd][cd] = deprocess_img[i][jd][cd]

            deprocess_img_mode = stats.mode(deprocess_img)
            deprocess_images_mode.append(deprocess_img_mode)
            plt.subplot(3,5,idx + 11)

            plt.imshow(overlay_img)

            overlay_images.append(overlay_img)


        plt.suptitle('layer: {}, neuron: #{}'.format( layer_name, neuron_num))

        plt.show()

        plt.clf()











        array = np.array(deconv[layer_name])


