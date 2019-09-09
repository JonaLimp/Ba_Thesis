import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator



def get_dataloader(batch_size, num, valid, is_train, gpu, seed,label):


		# one-hot-encoding for labels
	if label == 'fine': 
		classes = 100
		(x_train, y_train), (x_test, y_test) = cifar100.load_data(label)
		y_train, y_test = one_hot_encoding(y_train,y_test,classes)
	elif label == 'coarse': 
		classes = 20 
		(x_train, y_train), (x_test, y_test) = cifar100.load_data(label)
		y_train, y_test = one_hot_encoding(y_train,y_test,classes)
	else:
		(x_train, y_train_fine), (x_test, y_test_fine) = cifar100.load_data('fine')
		(x_train, y_train_coarse), (x_test, y_test_coarse) = cifar100.load_data('coarse')
		y_train,  y_test = one_hot_encoding(y_train_fine,y_test_fine,100)
		y_train, y_test = one_hot_encoding(y_train_coarse,y_test_coarse,20)



	#reshape dataset to have three channels:
	width, height, channels = x_train.shape[1], x_train.shape[2], 3
	x_train = x_train.reshape((x_train.shape[0],width,height,channels))

	width, height, channels = x_test.shape[1], x_test.shape[2], 3
	x_test = x_test.reshape((x_test.shape[0],width,height,channels))


	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	x_train /= 255.0
	x_test /= 255.0




	# number of validation samples
	num_val_samples = int(x_train.shape[0] * 0.2)


	x_val = x_train[:num_val_samples]
	partial_x_train= x_train[num_val_samples:]

	y_val = y_train[:num_val_samples]
	partial_y_train = y_train[num_val_samples:]

def one_hot_encoding(y_train,y_test,classes):
	
		y_train = keras.utils.to_categorical(y_train, classes)
		y_test = keras.utils.to_categorical(y_test, classes)
		return y_train, y_test


	

get_dataloader(32,10,0.2,10,1,42,'fine')