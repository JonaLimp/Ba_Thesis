from keras.applications.vgg16 import preprocess_input #TODO maybe not needed
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import layers
from keras.models import Model
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

def load_model(weights_path,data_shape):
    """
    Load and compile VGG model
    args: weights_path (str) trained weights file path
    returns model (Keras model)
    """
    # either VGG16() or VGG16_keras

    model = VGG_16_keras(weights_path,data_shape)
    model.compile(optimizer="sgd", loss='categorical_crossentropy')

    return model

def one_hot_encoding(y_train, y_test, classes):

    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    return y_train, y_test


def get_act_list(layer, data, top):
    
    act_list = []
    
    for sample in range(data[0]):
        sys.stdout.write("\rProcessing sample %s/%s" %
                         (sample + 1, len(data[0])))
        sys.stdout.flush()

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

def load_data():
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

    data = x_test[:50]
    data_shape = data.shape

    return data, data_shape

if __name__ == '__main__':
	
    data, data_shape = load_data()
    model = load_model(data_shape,'./Data/tester')
    model.summary()



    data_set = False 
    user_input = True
    all_images = False


    #in_array = np.expand_dims(image.img_to_array(data[0][0]), axis=0)
    #img_array = preprocess_input(in_array)
    #data = np.expand_dims(data, axis=0)
    #deconv = utils.visualize_all_layers(model, data[0])

    if data_set == True:

        data = np.expand_dims(data, axis=0)
        deconv = utils.visualize_all_layers(model, data[0])        
        plt.figure(figsize=(6 , 6))
        layer_name = 'block4_conv3'
        plt.suptitle(layer_name)
        content = np.array(deconv[layer_name])
        img = deconv[layer_name][1][0]
        img2 = deconv[layer_name][1][1]
        plt.subplot(2,2,1)
        plt.imshow(deprocess_image( img))

        plt.subplot(2,2,2)
        plt.imshow(data[0][0])

        plt.subplot(2,2,3)
        plt.imshow(deprocess_image(img2))

        plt.subplot(2,2,4)
        plt.imshow(data[0][1])
        plt.show()
        pdb.set_trace()

    elif user_input == True:

        data = np.expand_dims(data, axis=0)
        deconv = utils.visualize_all_layers(model, data[0])
        while True:

            img_num = input("insert image number: ")
            print(deconv.keys())
            layer_name = input("insert layer_name: ")
            print('image: #{}, "layer_name: {}'.format(img_num, layer_name))

            plt.figure(figsize=(6 , 6))
            
            plt.suptitle(layer_name)

            img = deconv[layer_name][0][int(img_num)]
            img2 = deconv[layer_name][1][int(img_num)]
            img3 = deconv[layer_name][2][int(img_num)]

            
            plt.subplot(2,2,1)
            plt.imshow(data[0][int(img_num)])

            plt.subplot(2,2,2)
            plt.imshow(deprocess_image(img))

            plt.subplot(2,2,3)
            plt.imshow(deprocess_image(img2))

            plt.subplot(2,2,4)
            plt.imshow(deprocess_image(img3))


            plt.show()
            array = np.array(deconv[layer_name])
            pdb.set_trace()

    elif all_images == True:
        data = np.expand_dims(data, axis=1)

        top_filter = []
        for idx in range(data.shape[0]):
            top_filter.append((utils.get_activations(model, data[idx]), idx))
        pdb.set_trace()










"""
for i, img in enumerate(deconv[layer_name]):
    plt.subplot(2,2,i+1)
    plt.imshow(deprocess_image(img))
    # Use the below commented line for block1 visualizations
    # Since, deprocess_image() is not required for block1 visualizations
    # plt.imshow(img)
    plt.title('Filter #{}'.format(i+1))
"""

"""

plt.imshow(deprocess_image(img))
plt.subplot(2,1,2)
plt.imshow(data[0])
plt.show()
"""
