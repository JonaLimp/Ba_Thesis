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

def create_model(type,img_shape,n_hidden,dropout,label):

	logger = logging.getLogger(__name__)

	if type == 'VGG16':

		pretrained = VGG16(weights='imagenet', include_top = False, input_shape = (img_shape[1],img_shape[2],img_shape[3]))

		network = layers.Flatten()(pretrained.output)
		network = layers.Dense(n_hidden, activation = 'relu')(network)
		network = layers.Dropout(dropout)(network)

	elif type == 'from_scratch':

		visible = layers.Input(shape = (img_shape[1],img_shape[2],img_shape[3]))

		network = layers.Conv2D(32, kernel_size=3,padding = 'same', activation='relu')(visible)
		network = layers.MaxPooling2D(pool_size=(2, 2))(network)
		network = layers.Dropout(dropout)(network)
		
		network = layers.Conv2D(32, kernel_size=3,padding = 'same', activation='relu')(network)
		network = layers.MaxPooling2D(pool_size=(2, 2))(network)
		network = layers.Dropout(dropout)(network)
		
		network = layers.Conv2D(64, kernel_size=3,padding = 'same', activation='relu')(network)
		network = layers.MaxPooling2D(pool_size=(2, 2))(network)
		network = layers.Dropout(dropout)(network)
		
		network = layers.Conv2D(64, kernel_size=3,padding = 'same', activation='relu')(network)
		network = layers.MaxPooling2D(pool_size=(2, 2))(network)
		network = layers.Dropout(dropout)(network)
		
		network = layers.Conv2D(128, kernel_size=3,padding = 'same', activation='relu')(network)
		network = layers.MaxPooling2D(pool_size=(2, 2))(network)
		network = layers.Dropout(dropout)(network)

		network = layers.Conv2D(128, kernel_size=3,padding = 'same', activation='relu')(network)
		network = layers.Dropout(dropout)(network)
		
		network = layers.Flatten()(network)
		network = layers.Dense(n_hidden, activation = 'relu')(network)
		network = layers.Dropout(dropout)(network)


	if label=='fine':
		out = layers.Dense(100,activation = 'softmax')(network)
	elif label=='coarse':
		out = layers.Dense(20,activation = 'softmax')(network)
	else:
		out_fine = layers.Dense(100,activation = 'softmax',name = 'fine')(network)
		out_coarse = layers.Dense(20,activation = 'softmax',name = 'coarse')(network)
		out = (out_fine, out_coarse)

	input_image = layers.Input(shape = (img_shape[1],img_shape[2],img_shape[3]))

	if type == 'fine':
		model = Model(inputs=pretrained.input, outputs=out)
	elif type == 'from_scratch':
		model = Model(inputs = visible, outputs=out)
	logger.info(model.summary())

	return model
