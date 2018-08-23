from base.base_train import BaseTrain
import tensorflow as tf


class TemplateTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader=None):
        super(TemplateTrainer, self).__init__(sess, model, config, logger, data_loader)

    def train_epoch(self, epoch=None):
        # implement the logic of epoch
        # loop on the number of iterations in the config and call the train step
        # add any summaries you want using the summary
        pass

    def train_step(self):
        # implement the logic of the train step
        # run the tensorflow session
        # return any metrics you need to summarize
        pass