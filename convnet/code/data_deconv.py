from keras.applications.vgg16 import preprocess_input #TODO maybe not needed
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation
from keras import layers
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import layers
from keras.models import Model
from keras import regularizers
from scipy import stats
from keras.applications.vgg16 import VGG16
import keras
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import keras.backend as K
import pdb
import convnet
import utils
import time
import pickle
import operator

def VGG_16_keras(data_shape,weights_path=None):

    visible = layers.Input(shape=(data_shape[1], data_shape[2], data_shape[3]))
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(visible)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    #x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    network = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)


    network = layers.Flatten()(network)

    network = layers.Dense(4096,activation = 'relu')(network)
    network = layers.Dense(4096,activation = 'relu')(network)
    network = layers.Dense(1024,activation = 'relu')(network)

    out = layers.Dense(100, activation="softmax")(network)

    model = Model(inputs=visible , outputs=out)




    model.summary()

    print("Loading weights...")
    model.load_weights(weights_path)

    return model

def BN_VGG(data_shape,weights_path=None):

    weight_decay = 0.5
    visible = layers.Input(shape=(data_shape[1], data_shape[2], data_shape[3]))

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_regularizer=regularizers.l2(weight_decay))(visible)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    #Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2), strides= (2,2), name ='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    network = layers.Dropout(0.5)(x)

    network = layers.Flatten()(x)

    network = layers.Dense(4096,activation = 'relu')(network)
    network = layers.Dense(4096,activation = 'relu')(network)
    network = layers.Dense(1024,activation = 'relu')(network)

    out = layers.Dense(100, activation="softmax")(network)

    model = Model(inputs=visible , outputs=out)




    model.summary()

    print("Loading weights...")
    model.load_weights(weights_path)

    layer_list = []


    for layer in model.layers:
        layer_list.append(layer)

    layer_list.pop(0)
    In = layers.Input(shape=(data_shape[1], data_shape[2], data_shape[3]))

    for idx,layer in enumerate(layer_list):


        if idx == 0:
            net = layer(In)
            continue

        if isinstance(layer, Conv2D):
            net = layer (net)

        elif isinstance(layer, MaxPooling2D):
            net = layer (net)

        elif isinstance(layer, Dense):
            net = layer (net)



    model = Model(inputs=In , outputs=net)

    print("Model without dropout and batchnorm layers")

    model.summary()

    #pdb.set_trace()

    return model

def load_model(weights_path,data_shape, model_type):
    """
    Load and compile VGG model
    args: weights_path (str) trained weights file path
    returns model (Keras model)
    """
    # either VGG16(), VGG16_keras or BN_VGG
    if model_type == 'VGG_16':
        model = VGG_16_keras(weights_path,data_shape)

    if model_type == 'BN_VGG':
        model = BN_VGG(weights_path,data_shape)

    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics = ["accuracy", "top_k_categorical_accuracy"])
    #pdb.set_trace()
    return model

def one_hot_encoding(y_train, y_test, classes):

    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    return y_train, y_test


def load_data(data_type):
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

    if data_type == 'act':

        data = x_test
        data_shape = data.shape

        return data, data_shape

    if data_type == 'test':
        return x_test, y_test

def get_layer_list(model):

    layer_list = []
    for idx in range(len(model.layers)-4):
        layer_list.append((model.layers[idx],model.layers[idx].name))

    return layer_list

    #TODO check for valid layers

def get_top_k_activation(model, data, k=5):

    #TODO improve performance by using activation model

    layer_outputs = [layer.output for layer in model.layers[1:] if isinstance(layer, Conv2D)]
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
            for f, feature_map in enumerate(layer[...,-1]):
                all_activations[idx,l,f] = feature_map.sum()

        activation_list.append(activations)
        print('sample:# ',idx)

    pdb.set_trace()


    indices = np.argpartition(all_activations, -k, axis=0)[-k:]
    top5_activations = []
    for i,idx in enumerate(indices):
        for l,layer in enumerate(idx):
            for f,maxidx in enumerate(layer):
                a = activation_list[maxidx][l]
                if f >= a.shape[-1]:
                    continue
                act = a[...,f]
                lname = layer_outputs[l].name
                top5_activations.append((act,(maxidx,lname,f,0,act.sum())))
    
    pdb.set_trace()

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
        if isinstance(layer, Conv2D):
            deconv_layers.append((convnet.DConvolution2D(layer),layer.name))
            deconv_layers.append((
                    convnet.DActivation(layer),layer.name + "_activation"))
        elif isinstance(layer, MaxPooling2D):
            deconv_layers.append((convnet.DPooling(layer),layer.name))
        elif isinstance(layer, Activation):

            deconv_layers.append((convnet.DActivation(layer),layer.name + "_activation"))
        elif isinstance(layer, keras.engine.input_layer.InputLayer):
            deconv_layers.append((convnet.DInput(layer),layer.name))


    return deconv_layers

def deconvolve_data(data, img_dict, layer_list):

    deconv_dict = {}

    deconv_layers = get_deconv_layer(layer_list)
    layer_idx = {}
    index = 0

    for layer in layer_list:

        if isinstance(layer[0], Conv2D):

            layer_idx.update({layer[1] : index})
            index += 1
            layer_idx.update({layer[1] + '_activation': index})
            index += 1


        else:
            layer_idx.update({layer[1] : index})
            index +=1




    data = np.expand_dims(data, axis=1)




    for sample in range(data.shape[0]):


        if sample in img_dict:
            print(sample)
            for index ,elem in enumerate(img_dict[sample]):



                print(elem)
                #print("Forward pass: sample #", sample, "layer: ", elem[1], "neuron: ", elem[2])

                deconv_layers[0][0].up([data[sample]])
                for i in range(1, len(deconv_layers)):

                    deconv_layers[i][0].up(deconv_layers[i - 1][0].up_data)
                    print(deconv_layers[i])


                output = deconv_layers[layer_idx[elem[1]]][0].up_data




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


                if isinstance(deconv_layers[layer_idx[elem[1]]][0].layer, Conv2D):

                    if elem[1] in deconv_dict.keys():
                            deconv_dict[elem[1]][elem[2]-1].append((elem,deconv))

                    else:

                        num_feature_maps = deconv_layers[layer_idx[elem[1]]][0].layer.get_weights()[0].shape[3]
                        deconv_dict.update({elem[1]: [ [] for x in range(num_feature_maps )]})

                        deconv_dict[elem[1]][elem[2]-1].append((elem, deconv))







    return deconv_dict



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





def get_activations(activation_save_path, layer_list, data, data_shape):

    top5 = get_top_k_activation(model, data)
    pickle.dump(top5, open(activation_save_path, 'wb'))
    print("Activation_dict saved")



def get_deconvolution(activation_save_path,deconv_save_path, data, layer_list):

    activation_dict = pickle.load(open(activation_save_path,'rb'))
    print("Activation_dict loaded")

    #img_dict = get_img_dict(activation_dict)
    #pdb.set_trace()
    #get_values(img_dict)

    deconv = deconvolve_data(data, activation_dict, layer_list)
    pickle.dump(deconv, open(deconv_save_path, 'wb'))
    print('deconvolved images are dumped')


def deconvolution_loop(deconv_save_path):

    while True:


        deconv = pickle.load(open(deconv_save_path,'rb'))
        pdb.set_trace()
        print(deconv.keys())
        #layer_name = input("Insert layer_name: ")
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
            overlay_img = data[neuron[idx][0][0]]

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


def load_deconv():


    deconv_save_path = '.convnet/Data/deconv_dict.pickle'
    deconv = pickle.load(open(deconv_save_path,'rb'))
    neuron = deconv['block1_conv1'][23][1]

    for key in deconv.keys():
        print(key)
        for idx ,elem in enumerate(deconv[key]):
            print(deconv[key][idx][0][0])

    #pdb.set_trace()


def test_model(model):

    model = load_model(data_shape,'./Data/tester')

    x_test, y_test = load_data('test')
    result = model.evaluate(x_test, y_test)
    print(results)



def visualize_neurons():

    deconv_save_path = './Data/deconv_dict.pickle'
    deconv = pickle.load(open(deconv_save_path,'rb'))

    #pdb.set_trace()
    key = 'block3_conv1'
    for idx in range(25):
        print(deconv[key][idx+5][1][0])
        plt.subplot(5,5,idx +1 )
        plt.imshow(deprocess_image( deconv[key][idx+5][1][1]))

    plt.show()

def get_highest_act(act_save_path):

    highest_act_list = []
    act_dict = pickle.load(open(act_save_path, 'rb'))
    for key  in act_dict.keys():
        for k in act_dict[key].keys():
            continue

    #pdb.set_trace()


if __name__ == '__main__':

    #model_load = False
    get_act = True
    get_deconv = True
    #load_deconv = False
    deconv_loop = False
    highest_act = False
    model_test = False
    #visualize_neurons = False

    data, data_shape = load_data('act')
    data_block1 = data[:500]
    data_name = 'data_block1'
    model_type = 'VGG_16'

    #./ckpt/Model with  Batch Normalization_#1 run BN_fine_LR=0.0001_HIDDEN_[4096, 4096, 1024]_BS=64_best_val_loss
    model = load_model(data_shape,'./ckpt/VGG16_miss_max_augmented_fine_#1 run one MPL missing, DA  _fine_LR=0.0001_HIDDEN_[4096, 4096, 1024]_BS=64', model_type)

    if model_test == True:
        test_model(model)

    layer_list = get_layer_list(model)
    #layer_list.pop(0)
    #layer_list =layer_list[:-10]
    #pdb.set_trace()
    activation_save_path = './convnet/Data/activation_dict_{}.pickle'.format(data_name)
    deconv_save_path = './convnet/Data/deconv_dict_{}.pickle'.format(data_name)


    # get activations for each neuron in each layer for given dataset
    # and save them as pickle file
    if get_act == True:
        get_activations(activation_save_path, layer_list, data_block1, data_shape)

    # get deconvs for each neuron in each layer for given dataset
    # and save them as pickle file
    if get_deconv == True:
        get_deconvolution(activation_save_path, deconv_save_path, data_block1, layer_list)

    if highest_act == True:
        get_highest_act(activation_save_path)

    #visualize specific neurons
    if deconv_loop == True:
        deconvolution_loop(deconv_save_path)












