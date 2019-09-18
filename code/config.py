"""
Config system
"""

from pathlib import Path
import numpy as np
from easydict import EasyDict as edict
import datetime
import logging
import sys

import pdb


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError("{} is not a valid config key".format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(
                ("Type mismatch ({} vs. {}) " "for config key: {}").format(
                    type(b[k]), type(v), k
                )
            )

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml

    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval

    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split(".")
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(
            d[subkey]
        ), "type {} does not match original type {}".format(
            type(value), type(d[subkey])
        )
        d[subkey] = value


def cfg_to_str(cfg):
    """Generate string representation of config"""
    return f"model_test"


def cfg_set_log_file(cfg):
    """Generate path to logfile from config"""
    if cfg.LOGGING.handlers.file.filename is None:
        log_dir = Path(cfg.LOGGING.LOG_DIR)
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        log_file = log_dir / (f"{cfg_to_str(cfg)}_{cfg.NAME}-" + cfg.LOGGING.LOG_FILE)
        cfg.LOGGING.handlers.file.filename = str(log_file)
    else:
        pass


__C = edict()

cfg = __C

#
# MISC options
#

# Note to log to remember run
__C.NOTE = ""

# Whether to use GPU
__C.GPU = False

# Random seed
__C.SEED = 42

# Directory in which model checkpoints are saved
__C.CKPT_DIR = "./ckpt"

# Directory in which Tensorboard logs are stored
__C.TENSORBOARD_DIR = "./tensorboard"

# Whether to use Tensorboard
__C.USE_TENSORBOARD = True

# Whether to resume training from checkpoint
__C.RESUME = False

# Name of the run, if empty then only model parameters define the name
__C.NAME = ""


#
__C.PRE_PROCESSING = edict()

# Which labels to use
__C.PRE_PROCESSING.LABEL = "fine"

#
# Training options
#
__C.TRAIN = edict()

# Whether to train the model or only test
__C.TRAIN.IS_TRAIN = False

# Minibatch size
__C.TRAIN.BATCH_SIZE = 32

# Number of epochs
__C.TRAIN.EPOCHS = 200

# Number of samples to use for training, if 0 then use full dataset
__C.TRAIN.NUM = 0

# Size of valudation split
__C.TRAIN.VALID = 0.1

# Number of samples to use for validation
__C.TRAIN.VAL_NUM = 0

# Inital learning rate value
__C.TRAIN.INIT_LR = 3e-4

# Number of epochs before reducing lr
__C.TRAIN.LR_PATIENCE = 10

# Number of epochs before earlystopping
__C.TRAIN.TRAIN_PATIENCE = 50

# Nesterov momentum value
__C.TRAIN.MOMENTUM = 0.9

# Whether to shuffle train and validation data
__C.TRAIN.SHUFFLE = True

# Optimizer to use for training {'sgd', 'adam'}
__C.TRAIN.OPTIM = "sgd"

# Amount of weight decay to use
__C.TRAIN.WEIGHT_DECAY = 0.0

# Decay factor for learning rate schedule
__C.TRAIN.SCHEDULE = [0.98]

# Loss function for reconstruction
__C.TRAIN.LOSS = "mse"

__C.TRAIN.VALID_FREQ = 1

__C.TRAIN.SAVE_PATH = "/my_model"
#
# Model options
#
__C.MODEL = edict()

__C.MODEL.TYPE = "default"

__C.MODEL.HIDDEN = 1024

__C.MODEL.DROPOUT = 0.5

__C.MODEL.BATCH_SIZE = 32


#
# Testing options
#

__C.TEST = edict()

# Number of samples to use for testing, if 0 then use full dataset
__C.TEST.NUM = 0
__C.TEST.LOAD_PATH = ""


#
# Logger Configuration
#

__C.LOGGING = edict()

__C.LOGGING.LOG_DIR = "./logs"

__C.LOGGING.LOG_FILE = f'{datetime.datetime.now().strftime("%Y-%m-%d:%H:%M")}.log'

__C.LOGGING.version = 1.0
__C.LOGGING.disable_existing_loggers = True

__C.LOGGING.loggers = edict()
__C.LOGGING.loggers[""] = edict()
__C.LOGGING.loggers[""].level = "INFO"
__C.LOGGING.loggers[""].handlers = ["console", "file"]

__C.LOGGING.formatters = edict()
__C.LOGGING.formatters.simple = edict()
__C.LOGGING.formatters.simple.format = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

__C.LOGGING.handlers = edict()

__C.LOGGING.handlers.console = edict()
__C.LOGGING.handlers.console["class"] = "logging.StreamHandler"
__C.LOGGING.handlers.console.level = "DEBUG"
__C.LOGGING.handlers.console.formatter = "simple"
__C.LOGGING.handlers.console.stream = "ext://sys.stdout"

__C.LOGGING.handlers.file = edict()
__C.LOGGING.handlers.file["class"] = "logging.FileHandler"
__C.LOGGING.handlers.file.filename = None
__C.LOGGING.handlers.file.level = "DEBUG"
__C.LOGGING.handlers.file.formatter = "simple"
