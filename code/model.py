import numpy as np
import pdb
import logging
import logging.config
import tensorflow as tf
from tensorflow import set_random_seed
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import logging


def create_model(type, pretrained, img_shape, n_hidden, dropout, label, arr_channels, VGG16_top, use_gen, dropout_arr, weight_decay):

    logger = logging.getLogger(__name__)

   
    # use VGG16 model for training
    if type == "VGG16":

        #either use pretrained model
        if pretrained == True:
            vgg16 = VGG16(
                weights="imagenet",
                include_top=VGG16_top,
                input_shape=(img_shape[1], img_shape[2], img_shape[3]),
            )

        #or initialize with random weights
        else:
            vgg16 = VGG16(
                weights = None,
                include_top=VGG16_top,
                input_shape=(img_shape[1], img_shape[2], img_shape[3]),
                )

        network = layers.Flatten()(vgg16.output)


        for i in range(len(n_hidden)):
            network = layers.Dense(n_hidden[i], activation="relu")(network)
            if dropout_arr[i]:
                network = layers.Dropout(dropout)(network)


        #network = layers.Dropout(dropout)(network)

    elif type == "VGG16_miss_Max":

        visible = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))
        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(visible)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

        network = layers.Flatten()(x)


        for i in range(len(n_hidden)):
            network = layers.Dense(n_hidden[i], activation="relu")(network)
            if dropout_arr[i]:
                network = layers.Dropout(dropout)(network)

    elif type == "VGG16_with_DO":

        visible = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))
        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(visible)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        network = layers.Flatten()(x)


        for i in range(len(n_hidden)):
            network = layers.Dense(n_hidden[i], activation="relu")(network)
            if dropout_arr[i]:
                network = layers.Dropout(dropout)(network)


    elif type == "less_pooling":

        visible = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))
        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_regularizer=regularizers.l2(weight_decay))(visible)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',kernel_regularizer=regularizers.l2(weight_decay))(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        network = layers.Flatten()(x)


        for i in range(len(n_hidden)):
            network = layers.Dense(n_hidden[i], activation="relu")(network)
            if dropout_arr[i]:
                network = layers.Dropout(dropout)(network)


    elif type =='VGG16_BN':

        visible = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))
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


        for i in range(len(n_hidden)):
            network = layers.Dense(n_hidden[i], activation="relu")(network)
            if dropout_arr[i]:
                network = layers.Dropout(dropout)(network)

    elif type == 'shallow_network_1':

        visible = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))
        # Block 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_regularizer=regularizers.l2(weight_decay))(visible)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Dropout(0.3)(x) 
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.MaxPooling2D((2,2), strides= (2,2), name ='block2_pool')(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        network = layers.Flatten()(x)


        for i in range(len(n_hidden)):
            network = layers.Dense(n_hidden[i], activation="relu")(network)
            if dropout_arr[i]:
                network = layers.Dropout(dropout)(network)

    elif type == 'shallow_network_2':

        visible = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))
        # Block 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_regularizer=regularizers.l2(weight_decay))(visible)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        x = layers.Dropout(0.3)(x) 

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2,2), strides= (2,2), name ='block2_pool')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        network = layers.Flatten()(x)


        for i in range(len(n_hidden)):
            network = layers.Dense(n_hidden[i], activation="relu")(network)
            if dropout_arr[i]:
                network = layers.Dropout(dropout)(network)
    
    elif type == 'shallow_network_3':

        visible = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))
        # Block 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_regularizer=regularizers.l2(weight_decay))(visible)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        x = layers.Dropout(0.3)(x) 

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2,2), strides= (2,2), name ='block2_pool')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
    
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        network = layers.Flatten()(x)


        for i in range(len(n_hidden)):
            network = layers.Dense(n_hidden[i], activation="relu")(network)
            if dropout_arr[i]:
                network = layers.Dropout(dropout)(network)
    
    elif type == 'kruger_net':

        visible = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))
        # Block 1
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_regularizer=regularizers.l2(weight_decay))(visible)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        x = layers.Dropout(0.25)(x) 

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2,2), strides= (2,2), name ='block2_pool')(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
    
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        x = layers.Dropout(0.25)(x)

        network = layers.Flatten()(x)


        for i in range(len(n_hidden)):
            network = layers.Dense(n_hidden[i], activation="relu")(network)
            if dropout_arr[i]:
                network = layers.Dropout(dropout)(network)

    elif type == 'shallow_net_less_dense':

        visible = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))
        # Block 1
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_regularizer=regularizers.l2(weight_decay))(visible)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        x = layers.Dropout(0.25)(x) 

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2,2), strides= (2,2), name ='block2_pool')(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
    
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        x = layers.Dropout(0.25)(x)

        network = layers.Flatten()(x)

        network = layers.Dense(1024, activation="relu")(network)
        network = layers.Dropout(0.5)(network)



    # use from scratch model
    # uses a model generator to create layerstacks consisting of
    # conV-, MaxPooling-, and Dropout layer

    elif type == 'kruger_net_nodrop':

        visible = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))
        # Block 1
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_regularizer=regularizers.l2(weight_decay))(visible)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)


        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.MaxPooling2D((2,2), strides= (2,2), name ='block2_pool')(x)


        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(weight_decay))(x)
    
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        x = layers.Dropout(0.25)(x)

        network = layers.Flatten()(x)


        for i in range(len(n_hidden)):
            network = layers.Dense(n_hidden[i], activation="relu")(network)
            if dropout_arr[i]:
                network = layers.Dropout(dropout)(network)
    # use from scratch model
    # uses a model generator to create layerstacks consisting of
    # conV-, MaxPooling-, and Dropout layer
    elif type == "from_scratch":
        if use_gen == True:

            visible = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))

            network = layers.Conv2D(
                arr_channels[0], kernel_size=3, padding="same", activation="relu"
            )(visible)
            network = layers.MaxPooling2D(pool_size=(2, 2))(network)
            network = layers.Dropout(dropout)(network)

            for i in range(len(arr_channels) - 2):

                network = layers.Conv2D(
                    arr_channels[i + 1], kernel_size=3, padding="same", activation="relu"
                )(network)
                #pdb.set_trace()
                network = layers.MaxPooling2D(pool_size=(2, 2))(network)
                network = layers.Dropout(dropout)(network)

            network = layers.Conv2D(
                arr_channels[-1], kernel_size=3, padding="same", activation="relu"
            )(network)
            network = layers.Dropout(dropout)(network)

            network = layers.Flatten()(network)

            #add FC-layers according to n_hidden
            for i in range(len(n_hidden)):
                network = layers.Dense(n_hidden[i], activation="relu")(network)

        #add model architecture here:
        else:

            model = VGG16(
                weights = None,
                include_top = VGG16_top,
                input_shape=(img_shape[1], img_shape[2], img_shape[3]))

            for layer in model.layers:
                print(layer)
            

        #network = layers.Dropout(dropout)(network)


    # implement classifier according to labeltype 
    if label == "fine":
        out = layers.Dense(100, activation="softmax")(network)
    elif label == "coarse":
        out = layers.Dense(20, activation="softmax")(network)
    else:
        out_fine = layers.Dense(100, activation="softmax", name="fine")(network)
        out_coarse = layers.Dense(20, activation="softmax", name="coarse")(network)
        out = (out_fine, out_coarse)

    input_image = layers.Input(shape=(img_shape[1], img_shape[2], img_shape[3]))

    if type == "VGG16":
        model = Model(inputs=vgg16.input, outputs=out)

    elif type == "VGG16_miss_Max":
        model = Model(inputs=visible, outputs=out)

    elif type == 'VGG16_BN':
        model = Model(inputs=visible, outputs=out)
    
    elif type == 'less_pooling':
        model = Model(inputs=visible, outputs=out)

    elif type == "from_scratch":
        model = Model(inputs=visible, outputs=out)

    elif type == "VGG16_with_DO":
        model = Model(inputs=visible, outputs=out)

    else:
        model = Model(inputs=visible, outputs=out)
    
    logger.info(model.summary())

    return model
