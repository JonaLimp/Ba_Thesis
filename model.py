import numpy as np
import logging
import logging.config
import tensorflow as tf
from tensorflow import set_random_seed
import tf.keras as keras
from keras import layers
from keras.applications.vgg16 import VGG16

def create_model(type,img_shape,n_hidden,dropout,label):

	network = VGG16(weights='imagenet', include_top = False,input_shape = (img_shape[1],img_shape[1],img_shape[3]))

	network = layers.Flatten(network)
	network = layers.Dense(n_hidden, activation = 'relu')(network)
	network = layers.Dropout(dropout)(network)
	
	if label=='fine':
		out = layers.Dense(100,activation = 'softmax')(network)
	elif label=='coarse':
		out = layers.Dense(20,activation = 'softmax')(network)
	else:
		out_fine = layers.Dense(100,activation = 'softmax')(network)
		out_coarse = layers.Dense(20,activation = 'softmax')(network)
		out = (out_fine, out_coarse)


	return out	
