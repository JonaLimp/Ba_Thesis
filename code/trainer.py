import numpy as np
import logging
from datetime import datetime
import logging.config
import tensorflow as tf
from tensorflow import set_random_seed
import tensorflow.keras as keras
from model import create_model
from pathlib import Path
import yaml


import pdb


class Trainer(object):
    """
    """

    def __init__(self, dataset, config, cnfg):
        """
        Construct a new Trainer instance.
        Params
        ----------------
        :param args         : (Object) object containing arguments.
        :param dataset  : (Object) dataset
        """

        #logger parmas
        self.logger = logging.getLogger(__name__)
        logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
        tf.logging.set_verbosity(tf.logging.INFO)

        # data sets and model params
        self.x_train, self.y_train = dataset["train"]
        self.x_test, self.y_test = dataset["test"]
        self.x_val, self.y_val = dataset["valid"]

        self.label = config.PRE_PROCESSING.LABEL
        shape = dataset["train"][0].shape

        self.model_name = f"{config.NAME}_{config.NOTE}_{config.PRE_PROCESSING.LABEL}_LR={config.TRAIN.INIT_LR}_HIDDEN_{config.MODEL.HIDDEN}_BS={config.TRAIN.BATCH_SIZE}"
        self.save_dir = Path(config.CKPT_DIR)
        self.save_path = self.save_dir / self.model_name
        file = open(config.YAML_DIR + "/" + self.model_name, "w")
        yaml.dump(cnfg, file)

        # create new model or load pretrained
        if not config.RESUME:
            self.model = create_model(
                config.MODEL.TYPE,
                config.MODEL.PRETRAINED,
                shape,
                config.MODEL.HIDDEN,
                config.MODEL.DROPOUT,
                config.PRE_PROCESSING.LABEL,
                config.MODEL.NEURON_ARR,
                config.MODEL.VGG16_top
            )
        else:
            self.model = tf.keras.models.load_model(self.save_path)

        if config.TRAIN.OPTIM == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                lr=config.TRAIN.INIT_LR, momentum=config.TRAIN.MOMENTUM
            )
        else:
            optimizer = tf.keras.optimizers.Adam(lr=config.TRAIN.INIT_LR)
        if self.label == "coarse" or self.label == "fine":
            self.model.compile(
                optimizer=optimizer,
                loss=config.TRAIN.LOSS,
                metrics=["accuracy", "top_k_categorical_accuracy"],
            )
        else:
            loss_dict = {"fine": config.TRAIN.LOSS, "coarse": config.TRAIN.LOSS}
            self.model.compile(
                optimizer=optimizer,
                loss=loss_dict,
                metrics=["accuracy", "top_k_categorical_accuracy"],
            )

        # train params
        self.batchsize = config.TRAIN.BATCH_SIZE
        self.epochs = config.TRAIN.EPOCHS
        self.lr = config.TRAIN.INIT_LR
        self.val_freq = config.TRAIN.VALID_FREQ
        self.train_patience = config.TRAIN.TRAIN_PATIENCE


        #tensorboard params


        self.tensor_dir = Path(config.TENSORBOARD_DIR) / self.model_name
        self.logger.info(f"[*] Saving tensorboard logs to {self.tensor_dir}")
        
        #overwrite existing tensorboards
        #if they are not retrained

        if not self.tensor_dir.exists():
            self.tensor_dir.mkdir(parents=True)
        else:
            if config.RESUME or not config.TRAIN.IS_TRAIN:
                pass
            else:
                for x in self.tensor_dir.iterdir():
                    if not x.is_dir():
                        x.unlink()
        self.file_writer = tf.summary.FileWriter(self.tensor_dir)

        #improvement params
        self.counter = 0
        self.best_val_loss = np.inf
        self.is_best = True

        # self.file_writer.set_as_default()

    def train(self):

        """
        Train the model on the training set.
        """


        layer_norms = np.zeros([self.epochs,len(self.model.layers)])
        is_best = False


        for epoch in range(self.epochs):

            self.logger.info(f"epoch: {epoch}")

            # fit single-label classifier
            if self.label == "fine" or self.label == "coarse":
                history = self.model.fit(
                    self.x_train,
                    self.y_train,
                    batch_size=self.batchsize,
                    validation_data=(self.x_val, self.y_val),
                    shuffle=True,
                    epochs=epoch + 1,
                    initial_epoch=epoch,
                    validation_freq=self.val_freq,
                )

            # fit multi-label classifier
            else:
                y_1_train, y_2_train = np.split(self.y_train, [100], 1)
                y_1_train = np.squeeze(y_1_train, axis=2)
                y_2_train = np.squeeze(y_2_train, axis=2)
                dict_y_train = {"fine": y_1_train, "coarse": y_2_train}

                y_1_val, y_2_val = np.split(self.y_val, [100], 1)
                y_1_val = np.squeeze(y_1_val, axis=2)
                y_2_val = np.squeeze(y_2_val, axis=2)
                dict_y_val = {"fine": y_1_val, "coarse": y_2_val}
                

                history = self.model.fit(
                    self.x_train,
                    dict_y_train,
                    batch_size=self.batchsize,
                    validation_data=(self.x_val, dict_y_val),
                    shuffle=True,
                    epochs=epoch + 1,
                    initial_epoch=epoch,
                    validation_freq=self.val_freq,
                )

            # print(history.history.keys())
            
            # check mean of layer weights
            for i, layer in enumerate(self.model.layers):
                if len(layer.get_weights()) > 0:

                    layer_norms[epoch, i] = np.mean(layer.get_weights()[0])
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        self.log_scalar(
                            "layer mean/layer {}".format(i), layer_norms[epoch, i], epoch
                    )


            # check for change in layer weights
            largest_change = layer_norms[epoch - 1] - layer_norms[epoch]
            for i in range(len(largest_change)):
                self.log_scalar(
                    "change of weights/layer {}".format(i), largest_change[i], epoch
                )

            #tensorboard scalars for single-label classifer
            if self.label == 'fine' or self.label == 'coarse':


                self.log_scalar("accuracy", history.history["acc"][0], epoch)
                self.log_scalar("loss", history.history["loss"][0], epoch)

                if not ((epoch + 1) % self.val_freq):
                    self.log_scalar(
                        "validation accuracy", history.history["val_acc"][0], epoch
                    )
                    self.log_scalar(
                        "validation loss", history.history["val_loss"][0], epoch
                    )


            #tensorboard scalars for multi-label classifier

            else:

                self.log_scalar("accuracy", history.history["fine_acc"][0], epoch)
                self.log_scalar("accuracy", history.history["coarse_acc"][0], epoch)
                self.log_scalar("loss", history.history["loss"][0], epoch)

                if not ((epoch + 1) % self.val_freq):
                    self.log_scalar(
                        "validation accuracy coarse",
                        history.history["val_coarse_acc"][0],
                        epoch,
                    )
                    self.log_scalar(
                        "validation accuracy fine",
                        history.history["val_fine_acc"][0],
                        epoch,
                    )
                    self.log_scalar(
                        "validation loss", history.history["val_loss"][0], epoch
                    )

            self.file_writer.flush()


            # pdb.set_trace()
            if not ((epoch + 1) % self.val_freq):
                is_best = self.best_val_loss < history.history["val_loss"][0]
                self.best_val_loss = min(
                    history.history["val_loss"][0], self.best_val_loss
                )
            else:
                is_best = False


            self.save_checkpoint(history, is_best)
            if not self.check_improvement(is_best):
                return







    def test(self, deconv = False):


        if self.label == "coarse" or self.label == "fine":
            score = self.model.evaluate(x=self.x_test, y=self.y_test)

        else:

            y_1_test , y_2_test = np.split(self.y_test,[100],1)
            y_1_test = np.squeeze(y_1_test,axis=2)
            y_2_test = np.squeeze(y_2_test,axis=2)
            dict_y_test = {'fine': y_1_test,'coarse':y_2_test}
            score = self.model.evaluate(x=self.x_test,y=dict_y_test)


    def deconv(self):

        pass


    def save_checkpoint(self, history, is_best):

        if is_best:
            self.model.save(Path(str(self.save_path) + "_best_val_loss"))

        self.model.save(self.save_path)

    def check_improvement(self, is_best):
        # check for improvement
        if not is_best:
            self.counter += 1
        else:
            self.counter = 0
        if self.counter > self.train_patience:
            self.logger.info("[!] No improvement in a while, stopping training.")
            return False
        return True

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
                Parameter
                ----------
                tag : basestring
                    Name of the scalar
                value
                step : int
                    training iteration
                """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.file_writer.add_summary(summary, step)
