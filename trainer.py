import numpy as np
import logging
import logging.config
import tensorflow as tf
from tensorflow import set_random_seed
import tf.keras as keras

import pdb


class Trainer(object):
    """
    """

    def __init__(self, dataset, args):
        """
        Construct a new Trainer instance.
        Params
        ----------------
        :param args         : (Object) object containing arguments.
        :param data_loader  : (Object) data iterator
        """

        self.logger = logging.getLogger(__name__)
        logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


    def train(self):



    def test(self):



