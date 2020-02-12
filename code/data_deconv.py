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

import pdb
import convnet
import utils
import time
import pickle
import operator
import copy
import os
import math

#just for testing VGG16 Representations

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import pandas as pd



# def BN_VGG(data_shape,weights_path=None):

#     weight_decay = 0.5
#     visible = layers.Input(shape=(data_shape[1], data_shape[2], data_shape[3]))

#     # Block 1
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_regularizer=regularizers.l2(weight_decay))(visible)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

#     #Block 2
#     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.4)(x)
#     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2,2), strides= (2,2), name ='block2_pool')(x)

#     # Block 3
#     x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.4)(x)
#     x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.4)(x)
#     x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

#     # Block 4
#     x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.4)(x)
#     x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.4)(x)
#     x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

#     # Block 5
#     x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.4)(x)
#     x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.4)(x)
#     x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=regularizers.l2(weight_decay))(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
#     network = layers.Dropout(0.5)(x)

#     network = layers.Flatten()(x)

#     network = layers.Dense(4096,activation = 'relu')(network)
#     network = layers.Dropout(0.5)(network)
#     network = layers.Dense(4096,activation = 'relu')(network)
#     network = layers.Dropout(0.5)(network)
#     network = layers.Dense(1024,activation = 'relu')(network)

#     out = layers.Dense(100, activation="softmax")(network)

#     model = Model(inputs=visible , outputs=out)




#     model.summary()
#     pdb.set_trace()

#     print("Loading weights...")
#     model.load_weights(weights_path)

    # layer_list = []


    # for layer in model.layers:
    #     layer_list.append(layer)

    # layer_list.pop(0)
    # In = layers.Input(shape=(data_shape[1], data_shape[2], data_shape[3]))

    # for idx,layer in enumerate(layer_list):


    #     if idx == 0:
    #         net = layer(In)
    #         continue

    #     if isinstance(layer, Conv2D):
    #         net = layer (net)

    #     elif isinstance(layer, MaxPooling2D):
    #         net = layer (net)

    #     elif isinstance(layer, Dense):
    #         net = layer (net)



    # model = Model(inputs=In , outputs=net)

    # print("Model without dropout and batchnorm layers")

   # model.summary()

    #pdb.set_trace()

   # return model

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

    deconv_dict = {}

    deconv_layers = get_deconv_layer(layer_list)
    layer_idx = {}
    index = 0


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


            if isinstance(deconv_layers[layer_idx[elem[1]]][0].layer, layers.Conv2D):

                if elem[1] in deconv_dict.keys():
                        deconv_dict[elem[1]][elem[2]-1].append((elem,deconv))

                else:

                    num_feature_maps = deconv_layers[layer_idx[elem[1]]][0].layer.get_weights()[0].shape[3]
                    deconv_dict.update({elem[1]: [ [] for x in range(num_feature_maps )]})

                    deconv_dict[elem[1]][elem[2]-1].append((elem, deconv))







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


def test_model(model):

    model = load_model(data_shape,'./Data/tester')

    x_test, y_test = load_data('test')
    result = model.evaluate(x_test, y_test)
    print(results)



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




def translate_representations(deconv_save_path,trans_rep_save_path, layer_list, model, data, num_neurons,steps):

    deconv = pickle.load(open(deconv_save_path,'rb'))
    layer_FWHM_dict = {}

    # n_list contains feature_maps
    for key, n_list in deconv.items():
        FWHM_list = []
        rand_neurons = np.random.choice(len(n_list), num_neurons, replace=False)
        pdb.set_trace()
        for neuron_idx in rand_neurons:

            neuron = n_list[neuron_idx][0]
            print(neuron[0])
            if not neuron[0][3]: #if activation is zero skip that neuron
                neuron = n_list[neuron_idx+1][0]
            # rep = data[neuron[0][0]] #at tuple position zero is image index
            rep = neuron[1] #at position 1 is the deconvolved representation
            act_array = shift_and_activate(rep,model,key,neuron_idx,steps)
            # new
            FWHM = getFWHM_GaussianFitScaledAmp(act_array)
            if not math.isnan(FWHM):
                FWHM_list.append(FWHM)
            # FWHM_list.append(getFWHM_GaussianFitScaledAmp(act_array))

            print(np.mean(FWHM_list))
        layer_FWHM_dict.update({key: np.nanmean(FWHM_list)})
        print('layer {} translated with {} randomly chosen neurons and {} steps.'.format(key,num_neurons,steps))

    pickle.dump(layer_FWHM_dict, open(trans_rep_save_path, 'wb'))
    print('transposed images are dumped')

    pdb.set_trace()
    return layer_FWHM_dict



def shift_and_activate(rep,model,layer,neuron_idx,steps):


    activation_model = Model(inputs=model.input, outputs=model.get_layer(layer).output)
    act_array = np.zeros((2*steps+1,2*steps+1))

    #rep[0][3] is the activation of the representation
    # act_array[steps,steps]= activation_model.predict()
    directions = ((0,1,0),(0,-1,0),(1,0,0),(-1,0,0),(-1,1,0),(1,-1,0),(-1,-1,0),(1,1,0))
    for direction in directions:
        for step in range(steps):
            cord = np.array(direction)*step

            input_rep = shift(rep,cord,mode='nearest')
            out = activation_model.predict(input_rep[None,...])
            act_array[cord[0]+steps,cord[1]+steps] = out[...,neuron_idx].sum()

    return act_array


# def VGG16_representation():

#     dir_path = './convnet/Data/VGG16/'

#     activation_save_path = os.path.join(dir_path, 'activation_dict_VGG.pickle')
#     deconv_save_path = os.path.join(dir_path, 'deconv_dict_VGG.pickle')
#     trans_rep_save_path = os.path.join(dir_path,'trans_rep_dict_VGG.pickle')
#     csv_save_path = os.path.join(dir_path, 'csv_VGG.pickle')
#     model = VGG16(weights='imagenet', include_top=True)
#     layer_list = get_pool_conv_layer_list(get_layer_list(model))
#     pdb.set_trace()
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)

#     img_path = os.path.join(os.getcwd(), 'convnet/cat.jpg')
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)

#     VGG_dict = get_top_k_activation(model, x, 1)

#     max_elem = None
#     max_value = 0.0


#     df_VGG = pd.DataFrame(VGG_dict[0], columns=['img','layer','neuron','activation'])
#     VGG_csv = df_VGG.to_csv()

#     for idx,elem in enumerate(VGG_dict[0]):
#         elem = list(elem)
#         string = elem[1]
#         elem[1] = string[:-2]
#         VGG_dict[0][idx] = tuple(elem)



    # VGG_deconv = deconvolve_data(x, VGG_dict, layer_list)

#/pickle.dump(deconv, open(deconv_save_path, 'wb'))

    # pickle.dump(VGG_dict, open(activation_save_path, 'wb'))
    # pickle.dump(VGG_deconv, open(deconv_save_path, 'wb'))
    # pickle.dump(VGG_csv, open(csv_save_path, 'wb'))

    # trans_rep = translate_representations(deconv_save_path, trans_rep_save_path, layer_list, model, 50, 5)



    # print('deconvolved images and csv are dumped')




if __name__ == '__main__':

    #model_load = False
    get_act = True
    get_deconv = True
    #load_deconv = False
    deconv_loop = False
    highest_act = False
    model_test = False
    trans_rep = False



    data, data_shape = load_data('act')
    #data = data[:100]
    #data_name = 'test_data'
    data_name = 'fine_data'
    model_type = 'code_BN_VGG'
    label = 'fine'

    weights_path = 'ckpt/Model with  Batch Normalization_#1 run BN_fine_LR=0.0001_HIDDEN_[4096, 4096, 1024]_BS=64_best_val_loss'
    # weights_path = 'ckpt/tester_test_fine_LR=0.03_HIDDEN_[2048, 2048]_BS=16_best_val_loss.h5'
    weights_path = os.path.join(os.getcwd(),weights_path)



    #./ckpt/VGG16_miss_max_augmented_fine_#1 run one MPL missing, DA  _fine_LR=0.0001_HIDDEN_[4096, 4096, 1024]_BS=64'
    model = load_model(data_shape,weights_path, model_type,label)

    if model_test == True:
        test_model(model)

    layer_list = get_layer_list(model)
    layer_list = get_pool_conv_layer_list(layer_list)
    #layer_list.pop(0)
    #layer_list =layer_list[:-10]
    #pdb.set_trace()
    dir_path = './convnet/Data/{}/'.format(data_name)
    # dir_path = './convnet/Data/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    activation_save_path = os.path.join(dir_path, 'activation_dict_{}.pickle'.format(data_name,data_name))
    deconv_save_path = os.path.join(dir_path, 'deconv_dict_{}.pickle'.format(data_name,data_name))
    trans_rep_save_path = os.path.join(dir_path,'trans_rep_dict_{}.pickle'.format(data_name,data_name))


    # get activations for each neuron in each layer for given dataset
    # and save them as pickle file
    if get_act == True:
        get_activations(activation_save_path, layer_list, data, data_shape)

    # get deconvs for each neuron in each layer for given dataset
    # and save them as pickle file
    if get_deconv == True:
        get_deconvolution(activation_save_path, deconv_save_path, data, layer_list)

    if highest_act == True:
        get_highest_act(activation_save_path)

    #visualize specific neurons
    if deconv_loop == True:
        deconvolution_loop(deconv_save_path)

    if trans_rep == True:
        translate_representations(deconv_save_path, trans_rep_save_path, layer_list, model, data, 60,15)











