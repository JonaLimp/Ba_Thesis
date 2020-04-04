import tensorflow as tf

#import tensorflow.keras as keras


#from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image



from scipy.ndimage import shift
import matplotlib.pyplot as plt
from utils import *
import numpy as np
from PIL import Image

import ipdb
import convnet
import utils
import time
import pickle
import operator
import copy
import os
import math
from save_and_compress import *
from plotting import *

from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)



def load_model(data_shape, weights_path, label):
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

    model = models.load_model(weights_path)

    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics = ["accuracy", "top_k_categorical_accuracy"])
    model.summary()

    return model

def get_layer_list(model):
    #retuns list of all layers in the model

    layer_list = []
    for idx in range(len(model.layers)-4):
        layer_list.append((model.layers[idx],model.layers[idx].name))

    return layer_list

    #TODO check for valid layers
def get_pool_conv_layer_list(layer_list):

    conv_pool_layer_list = []

    for layer in layer_list:


        if isinstance(layer[0], layers.Conv2D):

            conv_pool_layer_list.append(layer)


        if isinstance(layer[0], layers.MaxPooling2D):

            conv_pool_layer_list.append(layer)

    return conv_pool_layer_list

def get_top_k_activation(model, data, k):

    #TODO improve performance by using activation model
    #save all activations in an arrya of shape
    #dim #1 samples
    #dim #2 layer
    #dim #3 units

    layer_outputs = [layer.output for layer in model.layers[1:] if isinstance(layer, layers.Conv2D)]

    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    data = np.expand_dims(data, axis=1)

    activation_list = []

    start = time.clock()

    elapsed = time.clock()
    elapsed = elapsed - start


    all_activations = np.zeros((len(data),len(layer_outputs),int(layer_outputs[-1].shape[-1])))
    for idx ,sample in enumerate(data):
        activations = activation_model.predict(sample)
        for l,layer in enumerate(activations):
            for f in range(layer.shape[-1]):

                feature_map = layer[...,f]
                all_activations[idx,l,f] = np.abs(feature_map.sum())


    indices = np.argpartition(all_activations, -k, axis=0)[-k:]
    top5_activations = [[] for d in range(len(data))]
    for i,idx in enumerate(indices):
        for l,layer in enumerate(idx):
            for f,maxidx in enumerate(layer):
                if f >= layer_outputs[l].shape[-1]:
                    break
                lname = layer_outputs[l].name
                lname, _, _, = lname.partition('/')
                act_sum = all_activations[maxidx,l,f]

                top5_activations[maxidx].append((maxidx,lname,f,act_sum))


    elapsed = time.clock()
    elapsed = elapsed - start
    print("Time spent in (function name) is: ", elapsed)


    return top5_activations

def get_img_dict(activation_dict):

    img_dict = {}

    for layer in activation_dict.keys():
        for neuron in activation_dict[layer]:
            for tupl in activation_dict[layer][neuron]:

                act_tuple = (tupl[0], layer, neuron, tupl[1])
                if tupl[0] in img_dict.keys():
                    img_dict[tupl[0]].append(act_tuple)
                else:
                    img_dict.update({tupl[0]: [act_tuple] })

            activation_dict[layer][neuron].sort(key=operator.itemgetter(1))


    return img_dict

def get_deconv_layer(layer_list):

    counter = 0
    deconv_layers = []
    for layer in layer_list:


        layer = layer[0]
        if isinstance(layer, layers.Conv2D):
            deconv_layers.append((convnet.DConvolution2D(layer),layer.name))
            deconv_layers.append((
                    convnet.DActivation(layer),layer.name + "_activation"))
        elif isinstance(layer, layers.MaxPooling2D):
            deconv_layers.append((convnet.DPooling(layer),layer.name))
        elif isinstance(layer, layers.Activation):

            deconv_layers.append((convnet.DActivation(layer),layer.name + "_activation"))
        #TODO: check if istance(input) works
        elif isinstance(layer, keras.engine.input_layer.InputLayer):
            deconv_layers.append((convnet.DInput(layer),layer.name))


    return deconv_layers

def deconvolve_data(data, img_dict, layer_list):
    """Summary

    Args:
        data (TYPE): samples
        img_dict (TYPE): list containing index of neurons that shows a high activations for coressponding sample
        layer_list (TYPE): Description

    Returns:
        TYPE: Description
    """

    deconv_dict = {}

    deconv_layers = get_deconv_layer(layer_list)
    layer_idx = {}

    #help params to find zero arrays
    index = 0
    zero_deconv = []


    #get index
    for layer in layer_list:

        if isinstance(layer[0], layers.Conv2D):

            layer_idx.update({layer[1] : index})
            index += 1
            layer_idx.update({layer[1] + '_activation': index})
            index += 1


        else:
            layer_idx.update({layer[1] : index})
            index +=1


    data = np.expand_dims(data, axis=1)


    for sample in range(data.shape[0]):

            #print("Forward pass: sample #", sample, "layer: ", elem[1], "neuron: ", elem[2])

        #check if sample is not under top k samples
        if not len(img_dict[sample]):
            continue

        deconv_layers[0][0].up([data[sample]])
        for i in range(1, len(deconv_layers)):

            deconv_layers[i][0].up(deconv_layers[i - 1][0].up_data)
            print(deconv_layers[i])



        print(sample)
        for index ,elem in enumerate(img_dict[sample]):

            print(elem)

            output = deconv_layers[layer_idx[elem[1]]][0].up_data

            # if elem[2] > output.shape[-1]:
                # continue

            if output.ndim == 2:
                feature_map = output[:, elem[2]]
            else:
                feature_map = output[:, :, :, elem[2]]


            max_activation = feature_map.max()

            max_activation = feature_map.max()
            temp = feature_map == max_activation
            feature_map = feature_map * temp

            output_temp = np.zeros_like(output)

            if 2 == output.ndim:
                output_temp[:, elem[2]] = feature_map
            else:
                output_temp[:, :, :, elem[2]] = feature_map


            # Backward pass
            deconv_layers[layer_idx[elem[1]]][0].down(output_temp)
            for i in range(layer_idx[elem[1]]-1, - 1, -1):

                deconv_layers[i][0].down(deconv_layers[i + 1][0].down_data)


            deconv = deconv_layers[0][0].down_data
            deconv = deconv.squeeze()

            if not deconv.any():
                zero_deconv.append(elem)
                counter =+ 1
                print('zero again')


            if isinstance(deconv_layers[layer_idx[elem[1]]][0].layer, layers.Conv2D):

                if elem[1] in deconv_dict.keys():
                        deconv_dict[elem[1]][elem[2]].append((elem,deconv))

                else:

                    num_feature_maps = deconv_layers[layer_idx[elem[1]]][0].layer.get_weights()[0].shape[3]
                    deconv_dict.update({elem[1]: [ [] for x in range(num_feature_maps )]})

                    deconv_dict[elem[1]][elem[2]].append((elem, deconv))





                # if elem[1] in deconv_dict.keys():
                #         deconv_dict[elem[1]][elem[2]-1].append((elem,deconv))

                # else:

                #     num_feature_maps = deconv_layers[layer_idx[elem[1]]][0].layer.get_weights()[0].shape[3]
                #     deconv_dict.update({elem[1]: [ [] for x in range(num_feature_maps )]})

                #     deconv_dict[elem[1]][elem[2]-1].append((elem, deconv))
                #     pdb.set_trace()







    return deconv_dict






def get_activations(activation_save_path, layer_list, data, data_shape):

    top5 = get_top_k_activation(model, data, 5)
    pickle.dump(top5, open(activation_save_path, 'wb'))
    print("Activation_dict saved")



def get_deconvolution(activation_save_path,deconv_save_path, data, layer_list):

    activation_dict = pickle.load(open(activation_save_path,'rb'))
    print("Activation_dict loaded")

    #img_dict = get_img_dict(activation_dict)

    #get_values(img_dict)
    deconv = deconvolve_data(data, activation_dict, layer_list)
    pickle.dump(deconv, open(deconv_save_path, 'wb'))
    print('deconvolved images are dumped')


# def deconvolution_loop(deconv_save_path):

#     while True:


#         deconv = pickle.load(open(deconv_save_path,'rb'))
#         print(deconv.keys())
#         layer_name = input("Insert layer_name: ")
#         #layer_name = 'block1_conv2'
#         print ('There are {} units in layer {}'.format(len(deconv[layer_name]), layer_name))
#         neuron_num = input("Insert unit number: ")
#         #neuron_num = 42
#         print('layer_name: {}, neuron: #{}'.format(layer_name, neuron_num))

#         neuron = deconv[layer_name][int(neuron_num)]

#         deconv_img = []
#         img_list = []
#         overlay_images = []
#         deprocess_images_mode = []

#         plt.figure(figsize=(10 , 10))

#         for idx in range(5):
#             deconv_img.append(deprocess_image(neuron[idx][1]))
#             img_list.append(neuron[idx][0])
#             plt.subplot(3,5,idx+1)
#             plt.imshow(deprocess_image(neuron[idx][1]))


#         print("läuft")
#         for idx in range(5):

#             plt.subplot(3,5,idx + 6)
#             plt.imshow(data[neuron[idx][0][0]])


#         for idx in range(5):

#             deprocess_img = deprocess_image(neuron[idx][1])
#             overlay_img = copy.deepcopy(data[neuron[idx][0][0]])

#             for i in range(deprocess_img.shape[0]):
#                 for jd in range(deprocess_img.shape[1]):
#                     for cd in range(deprocess_img.shape[2]):
#                         #print(stats.mode(deprocess_img, axis=None)[0])
#                         if deprocess_img[i][jd][cd] <= 119 or deprocess_img[i][jd][cd] >= 128:
#                             overlay_img[i][jd][cd] = deprocess_img[i][jd][cd]

#             deprocess_img_mode = stats.mode(deprocess_img)
#             deprocess_images_mode.append(deprocess_img_mode)
#             plt.subplot(3,5,idx + 11)

#             plt.imshow(overlay_img)

#             overlay_images.append(overlay_img)


#         plt.suptitle('layer: {}, neuron: #{}'.format( layer_name, neuron_num))

#         plt.show()

#         plt.clf()











#         array = np.array(deconv[layer_name])


# def load_deconv():


#     deconv_save_path = '.convnet/Data/deconv_dict.pickle'
#     deconv = pickle.load(open(deconv_save_path,'rb'))
#     neuron = deconv['block1_conv1'][23][1]

#     for key in deconv.keys():
#         print(key)
#         for idx ,elem in enumerate(deconv[key]):
#             print(deconv[key][idx][0][0])

#     #pdb.set_trace()


def test_model(model,label):



    x_test, y_test = load_data('test',label)
    results = model.evaluate(x_test, y_test)
    print(results)
    ipdb.set_trace()



# def visualize_neurons():

#     deconv_save_path = './convnet/Data/VGG16/deconv_dict_VGG.pickle'
#     deconv = pickle.load(open(deconv_save_path,'rb'))


#     key = 'block5_conv3'
#     pdb.set_trace()
#     plt.imshow(deprocess_image(deconv[key][14][0][1]))
#     # pdb.set_trace()
#     # for key in deconv.keys():
#     #     deconv[key]
#     #     for idx in range(len(deconv.keys())):
#     #         plt.subplot(4,4,idx +1 )
#     #         plt.imshow(deprocess_image( deconv[key][0][0][1]))

#     plt.show()

# def get_highest_act(act_save_path):

#     highest_act_list = []
#     act_dict = pickle.load(open(act_save_path, 'rb'))
#     for key  in act_dict.keys():
#         for k in act_dict[key].keys():
#             continue

#     #pdb.set_trace()




def translate_representations(deconv_save_path, layer_list, model, num_neurons,steps,label,data):

    # trans_rep_save_path = os.path.join(dir_path,'trans_rep_dict_{}_num_neurons:{}_steps:{}_trans_input_type:{}.pickle'.format(data_name,num_neurons,steps,trans_input_type))
    deconv = pickle.load(open(deconv_save_path,'rb'))


    layer_FWHM_dict = {}
    layer_linreg_dict = {}
    neuron_characteristics_dict = {}

    # n_list contains feature_maps
    for key, n_list in deconv.items():

        #list to save the parameter of the different statistical methods
        FWHM_list = []
        linreg_list = []

        #list to save the values seperateed for every neuron
        neuron_characteristics_list = []


        #pick n randomly chosen neurons
        rand_neurons = np.random.choice(len(n_list), num_neurons, replace=False)
        #save all other neurons in seperated list
        other_neurons = [x for x in range(len(n_list)) if x not in rand_neurons]



        for neuron_idx in rand_neurons:

            neuron = n_list[neuron_idx][0]
            if not neuron[0][3]: #if activation is zero skip that neuron
                # neuron = n_list[neuron_idx+1][0]
                continue
            #sample, layer, fm, activation
            neuron_characteristics = neuron[0]
            print(neuron[0])

            # if trans_input_type == 'samples':
            # rep = data[neuron[0][0]] #at tuple position zero is image index
            # else:
            # rep = deprocess_image(neuron[1])#at position 1 is the deconvolved representation / adjust size
            rep = neuron[1]*255
            #ipdb.set_trace()
            act_array = shift_and_activate(rep,model,key,neuron_idx,steps)

            if not np.any(act_array):
                act_array, other_neurons, neuron_characteristics = shift_non_zeros(n_list, model, key, steps, other_neurons)
            if not np.any(act_array):
                ipdb.set_trace()


            #FWHM of Guassian
            #slope of regreseeion
            FWHM_all_units = []
            linreg_all_units = []

            #for elem in act_array:
            #    FWHM_all_units.append(getFWHM_GaussianFitScaledAmp(elem))
            #    linreg_all_units.append(get_linreg(elem,steps))
            FWHM = getFWHM_GaussianFitScaledAmp(act_array) #np.nanmean(FWHM_all_units)
            linreg = get_linreg_dir(act_array,steps)
 
            #np.nanmean(linreg_all_units)
            linreg_list.append(linreg)
            FWHM_list.append(FWHM)

            # FWHM_list.append(getFWHM_GaussianFitScaledAmp(act_array))
            res_and_index = [neuron_characteristics,linreg,FWHM,act_array]
            neuron_characteristics_list.append(res_and_index)

            print('FWHM of representation: {} and actual mean: {}'.format(FWHM, np.mean(FWHM_list)))
            print('Slope of representation: {} and actual mean: {}'.format(linreg, np.mean(linreg_list)))


        layer_linreg_dict.update({key: np.nanmean(linreg_list)})
        layer_FWHM_dict.update({key: np.nanmean(FWHM_list)})
        neuron_characteristics_dict.update({key: neuron_characteristics_list})

        print('layer {} translated with {} randomly chosen neurons and {} steps.'.format(key,num_neurons,steps))

    res = [layer_FWHM_dict,layer_linreg_dict,neuron_characteristics_dict]

    trans_rep_save_path = os.path.join(dir_path,'trans_rep_all_steps_{}.pickle'.format(label))
    ipdb.set_trace()
    pickle.dump(res, open(trans_rep_save_path, 'wb'))
    print('translated images are dumped')
    return res


def shift_and_activate(rep,model,layer,neuron_idx,steps):


    activation_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)


    heights, widths = model.get_layer(layer).output.shape[1],model.get_layer(layer).output.shape[2]

    act_array = np.zeros((2*steps+1,2*steps+1))

    #rep[0][3] is the activation of the representation
    center = activation_model.predict(rep[None,...])[...,neuron_idx].flatten()
    max_idx = np.argmax(center,0)
    act_array[steps,steps] = center[max_idx]

    directions = ((0,1,0),(0,-1,0),(1,0,0),(-1,0,0),(-1,1,0),(1,-1,0),(-1,-1,0),(1,1,0))
    for direction in directions:
        for step in range(1,steps+1):
            cord = np.array(direction)*step

            input_rep = shift(rep,cord,mode='constant')
            out = activation_model.predict(input_rep[None,...])
            act_array[cord[0]+steps,cord[1]+steps] = out[...,neuron_idx].flatten()[max_idx]
    #if layer == 'block1_conv1':
    #    ipdb.set_trace()
    return act_array



def shift_non_zeros(n_list,model,key,steps, other_neurons):

    neuron_idx = np.random.choice(other_neurons)
    other_neurons.remove(neuron_idx)
    neuron = n_list[neuron_idx][0]


    if not neuron[0][3]: #if activation is zero skip that neuron
        neuron_idx = np.random.choice(other_neurons)
        other_neurons.remove(neuron_idx)
        neuron = n_list[neuron_idx][0]

    neuron_characteristics = neuron[0]
    print(neuron_characteristics)
    rep = neuron[1]

    act_array = shift_and_activate(rep,model,key,neuron_idx,steps)
    if not np.any(act_array):
        act_array, other_neurons, neuron_characteristics= shift_non_zeros(n_list,model,key,steps,other_neurons)



    return act_array, other_neurons, neuron_characteristics



def translate_diff_steps_size(deconv_save_path,dir_path,layer_list, model,num_neurons,steps,label):

    steps += 1

    trans_rep_save_path = os.path.join(dir_path,'trans_rep_all_steps_{}.pickle'.format(label))
    res_list = []
    for step in range(2,steps+1):
        res = translate_representations(deconv_save_path,layer_list, model,num_neurons,step)
        # FWHM_list = [item for item in layer_FWHM_dict.items()]
        # FWHM_list.sort(key=lambda x:x[0])
        res_list.append(res)

    ipdb.set_trace()
    pickle.dump(res_list, open(trans_rep_save_path, 'wb'))
    print('translated images are dumped')
    # for idx, elem in enumerate(FWHM_steps_list):
    #     print('stepsize: {} \n'.format(idx+1))
    #     for e in elem:
    #         print (e)

    #     print('\n')




def calculate_kolmogrov(dir_path, deconv_save_path,label):

    kolmogrov_save_path = os.path.join(dir_path,'kolmogrov_mean_{}.pickle'.format(label))
    kolmogrov_mean_dict = save_and_compress_images(dir_path,deconv_save_path)
    pickle.dump(kolmogrov_mean_dict, open(kolmogrov_save_path, 'wb'))
    print('kolmogrov complexty dictionary dumped')


if __name__ == '__main__':
    np.random.seed(42)
    model_test = False
    get_act = False
    get_deconv = False


    trans_rep = True
    kolmogrov_complex = False
    create_individual_plots = True
    create_combined_plots = True

    # input for translation, either samples or deconvolved representations
    label = 'coarse'
    num_neurons = 128
    steps = 10





    #Set different data_name for different models
    # data_name = 'coarse_data'
    # model_type = 'code_BN_VGG'
    # label = 'coarse'

    #Set different data_name for different models

    #Model with fine label
    #weights_path = 'ckpt/Model_with_Batch_Normalization/Model_with_Batch_Normalization_Max_epochs:_fine_LR=0.0001_HIDDEN_[4096, 4096, 1024]_BS=64_best_val_loss.h5'

    #Model with coarse label
    # weights_path = 'ckpt//Model_with_Batch_Normalization_coarse_label/Model_with_Batch_Normalization_coarse_label_Epochs:300_LR=1.e-4_coarse_LR=0.0001_HIDDEN_[4096, 4096, 1024]_BS=64_best_val_loss.h5'
    # data_name = 'coarse_data'
    # model_type = 'code_BN_VGG'
    # label = 'coarse'

    #Model with fine labelling
    # weights_path = 'ckpt/shallow_network_4/shallow_network_4_Andrew_Kruger_fine_LR=0.0001_HIDDEN_[4096, 4096, 1024]_BS=64_best_val_loss.h5'
    # label = 'fine'

    #shallow_less_dense
    if label == 'fine':
        weights_path = 'ckpt/shallow_less_dense/shallow_less_dense_#3_fine_LR=0.0001_HIDDEN_[4096, 4096, 1024]_BS=64_best_val_loss.h5'
    else:
        weights_path ='ckpt/shallow_less_dense_coarse/shallow_less_dense_coarse_#3_coarse_LR=0.0001_HIDDEN_[4096, 4096, 1024]_BS=64_best_val_loss.h5'

    weights_path = os.path.join(os.getcwd(),weights_path)

    data, data_shape = load_data('act',label)
    # data = data[:6]
    #data_name = 'test_data'

    #./ckpt/VGG16_miss_max_augmented_fine_#1 run one MPL missing, DA  _fine_LR=0.0001_HIDDEN_[4096, 4096, 1024]_BS=64'
    model = load_model(data_shape,weights_path,label)


    if model_test == True:
        test_model(model,label)

    layer_list = get_layer_list(model)
    layer_list = get_pool_conv_layer_list(layer_list)

    dir_path = './results/{}/'.format(label)
    # dir_path = './convnet/Data/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    activation_save_path = os.path.join(dir_path, 'activation_dict_{}.pickle'.format(label))
    deconv_save_path = os.path.join(dir_path, 'deconv_dict_{}.pickle'.format(label))
    # kolmogrov_save_path =os.path.join(dir_path, 'kol_mean_{}.pickle'.format(label))


    # get activations for each neuron in each layer for given dataset
    # and save them as pickle file
    if get_act == True:
        get_activations(activation_save_path, layer_list, data, data_shape)

    # get deconvs for each neuron in each layer for given dataset
    # and save them as pickle file
    if get_deconv == True:
        get_deconvolution(activation_save_path, deconv_save_path, data, layer_list)

    # if highest_act == True:
    #     get_highest_act(activation_save_path)

    # #visualize specific neurons
    # if deconv_loop == True:
    #     deconvolution_loop(deconv_save_path,data)

    if trans_rep == True:
        # translate_diff_steps_size(deconv_save_path,dir_path,layer_list, model,num_neurons,steps,label)
        res = translate_representations(deconv_save_path, layer_list, model, num_neurons,steps, label,data)

    if kolmogrov_complex == True:
        calculate_kolmogrov(dir_path,deconv_save_path,label)

    if create_individual_plots == True:
        plot_results(label,model, dir_path,data, deconv_save_path)

    if create_combined_plots == True:
        combined_plots()



    # translate_diff_steps_size(deconv_save_path,dir_path, data_name,layer_list, model, data, num_neurons,steps,trans_input_type)








