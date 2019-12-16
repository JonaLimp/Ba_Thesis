import sys
import pdb
import numpy as np
from keras.layers import Input, InputLayer, Flatten, Activation,Dense
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dense
from convnet import DInput, DConvolution2D, DPooling, DActivation

import operator

def find_top_filters(output, top=7):
    """
    find filter with highest activation in given layer
        args:
            output of given filter
            top: number of filters with highest activation
    """
    #pdb.set_trace()
    filter_sum = []
    for filter_index in range(output.shape[-1]):
        if output.ndim == 2:
            sum_value = np.sum(output[:, filter_index])
        else:
            sum_value = np.sum(output[:, :, :, filter_index])
        if sum_value > 0:
            filter_sum.append((filter_index, sum_value))
    filter_sum.sort(key=lambda x: x[1], reverse=True)
    #pdb.set_trace()
    #TODO: returns now all filters
    #return filter_sum[:top]
    return filter_sum

def check_for_valid_layers(layer_list,layer_name):

    deconv_layers = []

    for layer in layer_list:
        if isinstance(layer, Conv2D):
            deconv_layers.append((layer.name, DConvolution2D(layer)))
            deconv_layers.append((layer.name + '_activation', DActivation(layer)))
        elif isinstance(layer, MaxPool2D):
            deconv_layers.append((layer.name, DPooling(layer)))
        elif isinstance(layer, Dense):
            pass
            deconv_layers.append((layer.name + '_activation', DActivation(layer)))
        elif isinstance(layer, Activation):
            deconv_layers.append((layer.name, DActivation(layer)))
        elif isinstance(layer, Flatten):
            pass
        elif isinstance(layer, InputLayer):
            deconv_layers.append((layer.name, DInput(layer)))
        else:
            print('Cannot handle this type of layer')
            print(layer.get_config())
            sys.exit()
        if layer_name == layer.name:
            break

    return deconv_layers

def visualize_all_layers(model, data, layer_name ='prediction', visualize_mode='max'):
    '''
    function to visualize feature
    # Arguments
        model: Pre-trained model used to visualize data
        data: image to visualize
        layer_name: Name of layer to visualize
        feature_to_visualize: Features to visualize
        visualize_mode: Visualize mode, 'all' or 'max', 'max' will only pick 
                        the greates activation in a feature map and set others
                        to 0s, this will indicate which part fire the neuron 
                        most; 'all' will use all values in a feature map,
                        which will show what image the filter sees. For 
                        convolutional layers, There is difference between 
                        'all' and 'max', for Dense layer, they are the same
    # Returns
        The image reflecting feature
    '''
    deconv_layers = []
    # Stack layers
    #tuple of layer_name and corresponding layer
    layer_list = []
    for idx in range(len(model.layers)-15):
        layer_list.append(model.layers[idx])

    for layer in layer_list:
        if isinstance(layer, Conv2D):
            deconv_layers.append((layer.name, DConvolution2D(layer)))
            deconv_layers.append((layer.name + '_activation', DActivation(layer)))
        elif isinstance(layer, MaxPool2D):
            deconv_layers.append((layer.name, DPooling(layer)))
        elif isinstance(layer, Dense):
            pass
            deconv_layers.append((layer.name + '_activation', DActivation(layer)))
        elif isinstance(layer, Activation):
            deconv_layers.append((layer.name, DActivation(layer)))
        elif isinstance(layer, Flatten):
            pass
        elif isinstance(layer, InputLayer):
            deconv_layers.append((layer.name, DInput(layer)))
        else:
            print('Cannot handle this type of layer')
            print(layer.get_config())
            sys.exit()
        if layer_name == layer.name:
            break

    # Forward pass

    deconv_layers[0][1].up(data)
    for i in range(1, len(deconv_layers)):

        deconv_layers[i][1].up(deconv_layers[i - 1][1].up_data)

    # Selecting layers to visualize
    layers_to_visualize = []

    model_layers = set([layer.name for layer in model.layers])
    layers_to_visualize = [x for x, y in enumerate(deconv_layers) 
                           if y[0] in model_layers]
    layers_to_visualize.reverse()
    # Removing the input layer
    layers_to_visualize.pop()
    print('layers_to_visualize:', layers_to_visualize)
    #TODO: layer_list just to check content of layers_to_visualize --> delete
    layer_list = [(layer.name, idx) for idx,layer in enumerate(model.layers) if idx in layers_to_visualize]

    deconv_dict = dict()
    for i in layers_to_visualize:
        deconv_list = []
        output = deconv_layers[i][1].up_data

        pdb.set_trace()

        top_filters = find_top_filters(output)
        print('output.shape :', output.shape)
        print('deconv_layer:', deconv_layers[i][0])
        print('top_filters:', top_filters)
        for feature_to_visualize, sum_value in top_filters:
            assert output.ndim == 2 or output.ndim == 4
            if output.ndim == 2:
                feature_map = output[:, feature_to_visualize]
            else:
                feature_map = output[:, :, :, feature_to_visualize]
            if 'max' == visualize_mode:
                max_activation = feature_map.max()
                temp = feature_map == max_activation
                feature_map = feature_map * temp

            elif 'all' != visualize_mode:
                print('Illegal visualize mode')
                sys.exit()
            output_temp = np.zeros_like(output)
            if 2 == output.ndim:
                output_temp[:, feature_to_visualize] = feature_map
            else:
                output_temp[:, :, :, feature_to_visualize] = feature_map

            # Backward pass
            deconv_layers[i][1].down(output_temp)
            for j in range(i - 1, -1, -1):
                deconv_layers[j][1].down(deconv_layers[j + 1][1].down_data)
            deconv = deconv_layers[0][1].down_data
            deconv = deconv.squeeze()
            deconv_list.append(deconv)
        deconv_dict[deconv_layers[i][0]] = deconv_list
    
    return deconv_dict



def get_activations(model, data, layer_name ='prediction', visualize_mode='max'):
    '''
    function to visualize feature
    # Arguments
        model: Pre-trained model used to visualize data
        data: image to visualize
        layer_name: Name of layer to visualize
        feature_to_visualize: Features to visualize
        visualize_mode: Visualize mode, 'all' or 'max', 'max' will only pick 
                        the greates activation in a feature map and set others
                        to 0s, this will indicate which part fire the neuron 
                        most; 'all' will use all values in a feature map,
                        which will show what image the filter sees. For 
                        convolutional layers, There is difference between 
                        'all' and 'max', for Dense layer, they are the same
    # Returns
    # Returns
        The image reflecting feature

    '''

    # Stack layers
    #tuple of layer_name and corresponding layer

    layer_list = []
    for idx in range(len(model.layers)-15):
        layer_list.append(model.layers[idx])

    deconv_layers = check_for_valid_layers(layer_list, layer_name)

    # Forward pass

    deconv_layers[0][1].up(data)
    for i in range(1, len(deconv_layers)):

        deconv_layers[i][1].up(deconv_layers[i - 1][1].up_data)

    # Selecting layers to visualize
    layers_to_visualize = []

    model_layers = set([layer.name for layer in model.layers])
    layers_to_visualize = [x for x, y in enumerate(deconv_layers) 
                           if y[0] in model_layers]
    layers_to_visualize.reverse()
    # Removing the input layer
    layers_to_visualize.pop()
    print('layers_to_visualize:', layers_to_visualize)
    #TODO: layer_list just to check content of layers_to_visualize --> delete
    layer_list = [(layer.name, idx) for idx,layer in enumerate(model.layers) if idx in layers_to_visualize]

    activation_dict = dict()
    for i in layers_to_visualize:
        deconv_list = []
        output = deconv_layers[i][1].up_data

        top_filters = find_top_filters(output)
        top_filters.sort(key = operator.itemgetter(0))

        print('output.shape :', output.shape)
        print('deconv_layer:', deconv_layers[i][0])
        print('top_filters:', top_filters)

        activation_dict.update({layer_list[0]: top_filters})

