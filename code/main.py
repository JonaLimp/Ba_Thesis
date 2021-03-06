from dataloader import get_dataloader
from trainer import Trainer

from config import cfg, cfg_from_file, cfg_from_list, cfg_set_log_file
import argparse
import numpy as np
import pprint
import logging
import logging.config
import sys
import tensorflow
from tensorflow import set_random_seed
from easydict import EasyDict as edict
import yaml


import pdb

from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a network")
    parser.add_argument(
        "-u", "--use_gpu", dest="gpu", help="Use GPU", default=0, type=int
    )
    parser.add_argument(
        "-c",
        "--cfg",
        dest="cfg_file",
        help="optional config file",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-s",
        "--set",
        dest="set_cfgs",
        help="set config keys",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def main(config,cnfg):
    """
    Main
    """

    dataset = get_dataloader(
        config.TRAIN.NUM,
        config.TRAIN.VALID,
        config.TEST.NUM,
        config.PRE_PROCESSING.LABEL,
        config.DATA_AUGMENTATION.DATA_AUG
    )

    cfg_set_log_file(cfg)
    logging.config.dictConfig(cfg.LOGGING)
    logger = logging.getLogger(__name__)
    logger.debug("Calling trainer")
    # instantiate trainer
    trainer = Trainer(dataset, config, cnfg)

    logger.debug("Start training")
    # either train
    logger.info(f"is_train is: {config.TRAIN.IS_TRAIN}")
    if config.TRAIN.IS_TRAIN:
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test(config.TEST.DECONV)


if __name__ == "__main__":

    args = parse_args()

    print("Called with args:")
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
        with open(args.cfg_file, "r") as f:
            cnfg = yaml.load(f)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.gpu:
        cfg.GPU = True

    # logging
    cfg_set_log_file(cfg)
    logging.config.dictConfig(cfg.LOGGING)
    # pdb.set_trace()
    logger = logging.getLogger(__name__)

    logger.info("Using config:")
    logger.info(pprint.pformat(cfg))

    # fix the random seeds
    np.random.seed(cfg.SEED)

    if args.gpu:
        set_random_seed(cfg.SEED)

    main(cfg,cnfg)
