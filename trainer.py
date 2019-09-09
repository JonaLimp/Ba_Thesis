import numpy as np
import logging
import logging.config
import tensorflow as tf
from tensorflow import set_random_seed
import tf.keras as keras
from model import create_model


import pdb


class Trainer(object):
    """
    """

    def __init__(self, dataset, config):
        """
        Construct a new Trainer instance.
        Params
        ----------------
        :param args         : (Object) object containing arguments.
        :param data_loader  : (Object) data iterator
        """

        self.logger = logging.getLogger(__name__)
        logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
        

        self.x_train, self.y_train = dataset['train']
        self.x_test, self.y_test = dataset['test']
        self.x_val, self.y_val = dataset['valid']

        self.label = config.PRE_PROCESSING.LABEL
        shape = dataset['train'][0].shape

        self.model = create_model(config.MODEL.TYPE, shape, config.MODEL.HIDDEN, config.MODEL.DROPOUT, config.PRE_PROCESSING.LABEL)
        model.compile(optimizer=config.TRAIN.OPTIM,
              loss= config.TRAIN.LOSS,
              metrics=['accuracy', 'top_k_categorical_accuracy'])
        
        self.batchsize = config.MODEL.BATCH_SIZE
        self.epochs = config.MODEL.EPOCHS
        self.lr = config.TRAIN.INIT_LR
        self.val_freg = config.TRAIN.VALID_FREQ



    def train(self):

        layer_norms = np.zeros([self.epochs,len(self.model.layers)])
        for epoch in range(self.epochs):
            model.fit(x_train,y_train, batchsize=self.batchsize, validation_data = (self.x_val,self.y_val), shuffle=True, epochs=1, initial_epoch=epoch, validation_freq=self.val_freq)

            for i,layer in enumerate(self.model.layers):
                layer_norms[epoch,i] = np.mean(layer.get_weights()[0])
 




    def test(self):



