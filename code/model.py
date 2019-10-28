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
import logging


def create_model(type, pretrained, img_shape, n_hidden, dropout, label, arr_channels,VGG16_top):

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

        #network = layers.Dropout(dropout)(network)

    
    # use from scratch model
    # uses a model generator to create layerstacks consisting of
    # conV-, MaxPooling-, and Dropout layer

    elif type == "from_scratch":
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

    elif type == "from_scratch":
        model = Model(inputs=visible, outputs=out)
    
    logger.info(model.summary())

    return model
