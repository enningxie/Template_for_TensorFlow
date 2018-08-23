import tensorflow as tf

from data_loader.mnist_data_generator import MnistData
from models.mnist_model import MnistModel
from trainers.mnist_trainer import MnistTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run argments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as err:
        print("Missing or invalid arguments {}".format(err))
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data_loader = MnistData(config)

    # create an instance of the model you want
    model = MnistModel(config, data_loader)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = MnistTrainer(sess, model, config, logger, data_loader)
    # load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()