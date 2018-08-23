from base.base_train import BaseTrain
import tensorflow as tf
from tqdm import tqdm
import time
import numpy as np


class MnistTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader=None):
        super(MnistTrainer, self).__init__(sess, model, config, logger, data_loader)
        self.train_init = self.data_loader.train_init_op
        self.test_init = self.data_loader.test_init_op

    def train(self):
        '''
        this is the main loop of training
        Looping on the epochs
        :return:
        '''
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
            self.train_epoch()
            self.eval_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self, epoch=None):
        # implement the logic of epoch
        # loop on the number of iterations in the config and call the train step
        # add any summaries you want using the summary
        self.sess.run(self.train_init)
        start_time = time.time()
        losses = []
        accs = []
        try:
            while True:
                loss, acc = self.train_step()
                losses.append(loss)
                accs.append(acc)
        except tf.errors.OutOfRangeError:
            pass

        loss_ = np.mean(losses)
        acc_ = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)

        print("epoch {}: loss->{}, acc->{}, {:.2f} seconds.".format(cur_epoch, loss_, acc_, time.time() - start_time))

        summaries_dict = {
            'loss': loss_,
            'acc': acc_
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        # implement the logic of the train step
        # run the tensorflow session
        # return any metrics you need to summarize

        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy])
        return loss, acc

    def eval_epoch(self):
        self.sess.run(self.test_init)
        start_time = time.time()
        losses = []
        accs = []
        try:
            while True:
                loss, acc = self.eval_step()
                losses.append(loss)
                accs.append(acc)
        except tf.errors.OutOfRangeError:
            pass

        loss_ = np.mean(losses)
        acc_ = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)

        print("eval_loss->{}, eval_acc->{}, {:.2f} seconds.\n".format(loss_, acc_, time.time() - start_time))

        summaries_dict = {
            'eval_loss': loss_,
            'eval_acc': acc_
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def eval_step(self):
        # implement the logic of the train step
        # run the tensorflow session
        # return any metrics you need to summarize

        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy])
        return loss, acc