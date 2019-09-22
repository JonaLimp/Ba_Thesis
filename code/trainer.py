import numpy as np
import logging
from datetime import datetime
import logging.config
import tensorflow as tf
from model import create_model


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

        self.x_train, self.y_train = dataset["train"]
        self.x_test, self.y_test = dataset["test"]
        self.x_val, self.y_val = dataset["valid"]

        self.label = config.PRE_PROCESSING.LABEL
        shape = dataset["train"][0].shape

        self.model = create_model(
            config.MODEL.TYPE,
            shape,
            config.MODEL.HIDDEN,
            config.MODEL.DROPOUT,
            config.PRE_PROCESSING.LABEL,
        )

        if self.label == "coarse" or self.label == "fine":
            self.model.compile(
                optimizer=config.TRAIN.OPTIM,
                loss=config.TRAIN.LOSS,
                metrics=["accuracy", "top_k_categorical_accuracy"],
            )
        else:
            loss_dict = {"fine": config.TRAIN.LOSS, "coarse": config.TRAIN.LOSS}
            self.model.compile(
                optimizer=config.TRAIN.OPTIM,
                loss=loss_dict,
                metrics=["accuracy", "top_k_categorical_accuracy"],
            )

        self.batchsize = config.MODEL.BATCH_SIZE
        self.epochs = config.TRAIN.EPOCHS
        self.lr = config.TRAIN.INIT_LR
        self.val_freq = config.TRAIN.VALID_FREQ

        self.logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer = tf.summary.FileWriter(self.logdir + "/metrics")

        self.save_path = config.TRAIN.SAVE_PATH
        # self.file_writer.set_as_default()

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

    def train(self):

        layer_norms = np.zeros([self.epochs, len(self.model.layers)])

        for epoch in range(self.epochs):

            print("epoch: ", epoch)

            if self.label == "fine" or self.label == "coarse":
                history = self.model.fit(
                    self.x_train,
                    self.y_train,
                    batch_size=self.batchsize,
                    validation_data=(self.x_val, self.y_val),
                    shuffle=True,
                    epochs=1,
                    # initial_epoch=epoch,
                    validation_freq=self.val_freq,
                )

            else:

                y_1_train, y_2_train = np.split(self.y_train, [100], 1)
                y_1_train = np.squeeze(y_1_train, axis=2)
                y_2_train = np.squeeze(y_2_train, axis=2)
                dict_y_train = {"fine": y_1_train, "coarse": y_2_train}

                y_1_val, y_2_val = np.split(self.y_val, [100], 1)
                y_1_val = np.squeeze(y_1_val, axis=2)
                y_2_val = np.squeeze(y_2_val, axis=2)
                dict_y_val = {"fine": y_1_val, "coarse": y_2_val}
                # pdb.set_trace()

                history = self.model.fit(
                    self.x_train,
                    dict_y_train,
                    batch_size=self.batchsize,
                    validation_data=(self.x_val, dict_y_val),
                    shuffle=True,
                    epochs=1,
                    # initial_epoch=epoch,
                    validation_freq=self.val_freq,
                )
                # pdb.set_trace()
            print(history.history.keys())

            for i, layer in enumerate(self.model.layers):
                if len(layer.get_weights()) > 0:
                    layer_norms[epoch, i] = np.mean(layer.get_weights()[0])
                    self.log_scalar(
                        "layer norm/layer {}".format(i), layer_norms[epoch, i], epoch
                    )

            largest_change = layer_norms[epoch - 1] - layer_norms[epoch]
            for i in range(len(largest_change)):
                self.log_scalar(
                    "change of weights/layer {}".format(i), largest_change[i], epoch
                )

            if self.label == "fine" or self.label == "coarse":

                self.log_scalar(
                    "validation accuracy", history.history["val_acc"][0], epoch
                )
                self.log_scalar("accuracy", history.history["acc"][0], epoch)
                self.log_scalar("loss", history.history["loss"][0], epoch)
                self.log_scalar(
                    "validation loss", history.history["val_loss"][0], epoch
                )

            else:

                self.log_scalar(
                    "validation accuracy fine",
                    history.history["val_fine_acc"][0],
                    epoch,
                )
                self.log_scalar("accuracy", history.history["fine_acc"][0], epoch)

                self.log_scalar(
                    "validation accuracy coarse",
                    history.history["val_coarse_acc"][0],
                    epoch,
                )
                self.log_scalar("accuracy", history.history["coarse_acc"][0], epoch)

                self.log_scalar("loss", history.history["loss"][0], epoch)
                self.log_scalar(
                    "validation loss", history.history["val_loss"][0], epoch
                )

            self.file_writer.flush()

        self.model.save(self.save_path)

    def test(self):

        if self.label == "coarse" or self.label == "fine":
            score = self.model.evaluate(x=self.x_test, y=self.y_test)

        else:
            y_1_test, y_2_test = np.split(self.y_test, [100], 1)
            y_1_test = np.squeeze(y_1_test, axis=2)
            y_2_test = np.squeeze(y_2_test, axis=2)
            dict_y_test = {"fine": y_1_test, "coarse": y_2_test}
            score = self.model.evaluate(x=self.x_test, y=dict_y_test)
            # FIXME score should be used
            print(score)
