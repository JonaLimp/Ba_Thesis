import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Model
import keras.backend as K

import pdb

class DInput(object):
	
	def __init__(self, layer):
		"""
		#Arguments
			layers: an instance of Input layer, whose configuaration will be used to initiate DInput(input_shape,
			output_shape, weights)
		"""

		self.layer = layer

	#input and output of INput layer are the same

	def up(self, data):

		self.up_data = data
		return self.up_data

	def down(self, data):
		"""
		function to operate input in backward pass, the input and output are the same
		#Arguments
			data: data to be operated in backward pass
		#Returns
			data
		"""

		self.down_data = data
		return self.down_data

class DConvolution2D(object):

	def __init__(self,layer):

		"""
		#Arguments
			layer: Conv layer, whose configuration will be used to initiate  
			DConvolution2D (input_shape, output_shape, weights)
		"""

		self.layer = layer

		weights = layer.get_weights()
		W, b = weights
		config = layer.get_config()

		#Set up_func for DConV
		
		input_ = layers.Input(shape = layer.input_shape[1:])
		output = layers.Conv2D.from_config(config)(input_)
		up_func = Model(input_,output)
		up_func.layers[1].set_weights(weights)
		self.up_func = up_func
		
		#Flip W horizontally and vertically, and set down_func for convoluton

		W =np.transpose(W,(0,1,3,2))
		W + W[::-1, ::-1,:,:]

		config['filters'] = W.shape[3]
		config["kernel_size"] = (W.shape[0], W.shape[1])
		b = np.zeros(config['filters'])
		input_ = layers.Input(shape = layer.output_shape[1:])
		output = layers.Conv2D.from_config(config)(input_)
		down_func = Model(input_, output)
		down_func.layers[1].set_weights((W,b))
		self.down_func = down_func 
		
	def up(self,data):
		"""
		function to compute Conv output in forward pass
		#Arguments
			data: Data to be operated in forward pass
		#Returns
			Convolved result
		"""

		self.up_data = self.up_func.predict(data)

		return self.up_data

	def down(self, data):
		"""
		function to compute Deconv output in backward pass
		#Arguments
			data: Data to be operated in backward pass
		#Returns:
			Deconvovled result
		"""

		self.down_data = self.down_func.predict(data)
		return self.down_data

class DPooling(object):
	"""
	A Class to define forward and backward operation on Pooling

	"""

	def __init__(self, layer):
		"""
		#Arguments 
			layer: an instance of Pooling layer, whose configuration will be used to initiate 
			DPooling (input_shape, output_shape, weights)
		"""

		self.layer = layer
		self.pool_size = layer.pool_size

	def up(self, data):
		"""
		function to compute pooling output in forward pass
		#Arguments
			data: Data to be operated in forward pass
		#Returns
			Pooled result
		"""
		[self.up_data, self.switch] = self.__max_pooling_with_switch(data, self.pool_size)
		return self.up_data

	def down(self, data):
		"""
		function to compute unpooling output in backward pass
		# Arguments
			data: Data to be operated in forward pass
		#Returns 
			Unpooled result
		"""
		self.down_data = self.__max_unpooling_with_switch(data, self.switch)
		return self.down_data

	def __max_pooling_with_switch(self, input_, pool_size):
		"""
		Compute pooling and switch in forward pass, switch stores 
		location of the maximum value in each poolsize * poolsize block
		#Arguments
			input: data to be pooled
			poolsize: size of pooling operation
		# Returns
			Pooled result and Switch
		"""

		switch = np.zeros(input_.shape)
		out_shape = list(input_.shape)
		row_pool_size = int(pool_size[0])
		col_pool_size = int(pool_size[1])
		out_shape[1] = out_shape[1] // pool_size[0]
		out_shape[2] = out_shape[2] // pool_size[1]
		pooled = np.zeros(out_shape)

		for sample in range(input_.shape[0]):
			for dim in range(input_.shape[3]):
				for row in range(out_shape[1]):
					for col in range(out_shape[2]):

						patch = input_[sample,
										row * row_pool_size : (row + 1) * row_pool_size,
										col * col_pool_size : (col + 1) * col_pool_size, 
										dim ]
						max_value = patch.max()
						pooled[sample, row, col, dim] = max_value
						max_col_index = patch.argmax(axis = -1)
						max_cols = patch.max(axis = -1)
						max_row = max_cols.argmax()
						max_col = max_col_index[max_row]
						switch[sample,
								row * row_pool_size + max_row,
								col * col_pool_size + max_col,
								dim] = 1

		return [pooled, switch]


	def __max_unpooling_with_switch(self,input_, switch):
		"""
		Compute unpooled output using pooled data and sawitch 
		#arguments
			input: data to be pooled 
			pool_size: size of pooled operatin
			switch : storing location of each element
		# Returns
			Unpooled result
		"""

		out_shape = switch.shape
		unpooled = np.zeros(out_shape)
		for sample in range(input_.shape[0]):
			for dim in range(input_.shape[3]):
				tile = np.ones((switch.shape[1]//input_.shape[1],
					switch.shape[2]//input_.shape[2]))
				out = np.kron(input_[sample, :, :, dim], tile)
				unpooled[sample, :, :, dim] = out * switch[sample, :, :, dim]

		return unpooled

class DActivation(object):

	def __init__(self, layer, linear = False):
		"""
		#Arguments
			layer: an instance of Activation layer, whose configuration will be used to
			initiate DActivation (input_shape, outpur_shape, weights)
		"""
		print('got_called')
		self.layer = layer
		self.linear = linear
		self.activation = layer.activation

		input_ = K.placeholder(shape = layer.output_shape)


		output = self.activation(input_)

		#Do same activation in both pass

		self.up_func = K.function(
					[input_, K.learning_phase()], [output])

		self.down_func = K.function(
					[input_, K.learning_phase()], [output])



	def down(self, data, learning_phase = 0):
		self.down_data = self.down_func([data, learning_phase])[0]
		return self.down_data

	def up(self,data, learning_phase = 0):

		self.up_data = self.up_func([data,learning_phase])[0]
		return self.up_data
"""
if __name__ == '__main__':
	data_shape = (12,224,224,3)
	visible = layers.Input(shape=(data_shape[1], data_shape[2], data_shape[3]))
	layer = Conv2D(kernel_size = (3,3),filters=512,activation='relu',kernel_initializer='random_uniform', bias_initializer='zeros')(visible)
	out = layers.Dense(100, activation="softmax")(layer)
	model = Model(inputs=visible , outputs=out)
	model.compile(optimizer="sgd", loss='categorical_crossentropy')
	new_layer = DConvolution2D(model.layers[1])
"""
